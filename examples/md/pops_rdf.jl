using Pkg
Pkg.activate(@__DIR__)

using ACEpotentials, ACEfit
using POPS
using AtomsBuilder
using Molly
using Unitful
using LinearAlgebra, Random, Statistics
using CairoMakie
using ProgressMeter


# fit POPS ace linear model on Si tiny dataset

data_raw, _, _ = ACEpotentials.example_dataset("Si_tiny")

rng = Xoshiro(2026)

println("Silicon dataset: $(length(data_raw)) total structures")

Eref = [:Si => -158.54496821]
rcut = 4.0
model = ace1_model(; elements=[:Si], order=3, totaldegree=10, rcut=rcut, Eref=Eref)
n_basis = length(model.ps.WB) + length(model.ps.Wpair)

weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0)) # default weights
datakw = (energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
make_atomsdata(xs) = [ACEpotentials.AtomsData(s; weights=weights, v_ref=model.model.Vref, datakw...) for s in xs]
data = make_atomsdata(data_raw)

A, Y, W = ACEfit.assemble(data, model)
P = ACEpotentials.Models.algebraic_smoothness_prior(model.model; p=4)
Ap = Diagonal(W) * (A / P)
Yp = W .* Y

pops = fit(POPSModel, Ap, Yp;
    prior_covariance=1e-4,
    leverage_percentile=0.1)

println("fitted POPS ensemble: n_basis=$n_basis, effective rank R=$(length(pops.lower_bounds))")

θ0_pre = vec(pops.coef)           # coefficients in preconditioned basis
θ0_orig = P \ θ0_pre               # coefficients in native coordinates
ACEpotentials.Models.set_linear_parameters!(model, θ0_orig)


# Setup MD simulation


T_md = 300.0u"K"
sys0 = bulk(:Si, cubic=true) * (3, 3, 3) # 216 atoms, 16.29 Angstrom
rattle!(sys0, 0.03)

sys_md = Molly.System(sys0; force_units=u"eV/Å", energy_units=u"eV")

sys_md = Molly.System(
    sys_md;
    general_inters=(fast_evaluator(model),),
    velocities=Molly.random_velocities(sys_md, T_md)
)

simulator = Langevin(
    dt=1.0u"fs",
    temperature=T_md,
    friction=1.0u"ps^-1",) # BAOA integrator

println()

# Equilibrate system for 2 ps

println("equilibrating...")
Molly.simulate!(sys_md, simulator, 2_000)

# compute unnormalized pair distance histogram

function pair_histogram!(h::AbstractVector{<:Integer}, sys, edges)
    fill!(h, 0)
    coords = sys.coords
    boundary = sys.boundary
    rmax = edges[end]
    dx = step(edges)
    nb = length(h)
    n = length(coords)
    @inbounds for i in 1:n-1, j in i+1:n
        r = norm(Molly.vector(coords[i], coords[j], boundary))
        r ≥ rmax && continue
        b = clamp(Int(fld(ustrip(r), ustrip(dx))) + 1, 1, nb)
        h[b] += 1
    end
    return h
end

# Sample for 20 ps

n_frames = 200
chunk_steps = 100
n_bins = 300
r_max = 8.14u"Å"             # < L/2 ≈ 8.15 Å
edges = range(0.0u"Å", r_max, length=n_bins + 1)

ace_features_buf = zeros(n_basis, n_frames) # buffer for ACE energy features
histos_buf = zeros(Int, n_frames, n_bins) # buffer for pair histograms
h_buf = zeros(Int, n_bins)

println("production: $n_frames frames × $chunk_steps fs ...")
@showprogress for i in 1:n_frames
    Molly.simulate!(sys_md, simulator, chunk_steps)
    Xi = ACEpotentials.Models.potential_energy_basis(sys_md, model)  # eV vector
    ace_features_buf[:, i] .= ustrip.(u"eV", Xi)
    pair_histogram!(h_buf, sys_md, edges)
    histos_buf[i, :] .= h_buf
end

# compute rdf from pair distances histogram
function rdf_normalize(h_avg, edges, sys)
    N = length(sys.coords)
    Vbox = ustrip(u"Å^3", Molly.volume(sys))
    e = ustrip.(u"Å", collect(edges))
    n_pairs_ideal = N * (N - 1) / 2
    g = similar(h_avg, Float64)
    for b in eachindex(h_avg)
        shell = (4 / 3) * π * (e[b+1]^3 - e[b]^3)
        g[b] = h_avg[b] / (n_pairs_ideal * shell / Vbox)
    end
    return g
end

# reweight to POPS ensemble

n_samples = 2000 # number of posterior samples
S_pre = sample(pops, n_samples; sampling_method=:sobol)   # n_basis × n_samples
β = ustrip(u"eV^-1", 1 / (Unitful.k * T_md)) # 1/eV

Hf = Float64.(histos_buf) # n_frames × n_bins
g_samples = zeros(n_samples, n_bins) # rdfs

for s in 1:n_samples
    Δθ_orig = P \ (S_pre[:, s] .- θ0_pre) # apply inverse preconditioner
    ΔU = ace_features_buf' * Δθ_orig # compute energy difference accross frames

    # compute boltzmann reweighting factor
    logw = -β .* ΔU
    logw .-= maximum(logw) # for numerical stability
    w = exp.(logw)
    Z = sum(w)

    h_avg = vec((w' * Hf) ./ Z) # reweight pair histogram
    g_samples[s, :] .= rdf_normalize(h_avg, edges, sys_md) # compute corresponding RDF
end

g0 = rdf_normalize(vec(mean(Hf; dims=1)), edges, sys_md) # pops mean rdf
g_lo = vec(minimum(g_samples; dims=1)) # pops lower bound
g_hi = vec(maximum(g_samples; dims=1)) # pops upper bound

# plot results

r_centers = ustrip.(u"Å", 0.5 .* (edges[1:end-1] .+ edges[2:end]))

fig = Figure()
ax = Axis(fig[1, 1];
    xlabel="r [Å]",
    ylabel="g(r)",
    title="POPS bounds on Si RDF at 300K",
)
band!(ax, r_centers, g_lo, g_hi; color=(:blue, 0.3), label="min/max bounds (POPS)")
lines!(ax, r_centers, g0; color=:black, label="ridge MLIP", linewidth=0.5)
axislegend(ax; position=:rt)

out = joinpath(@__DIR__, "rdf_pops.png")
save(out, fig)
println("wrote $out")
