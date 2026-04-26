using Pkg
Pkg.activate(@__DIR__)

using ACEpotentials, ACEfit
using POPS
using LinearAlgebra, Random, Statistics

data, _, _ = ACEpotentials.example_dataset("Si_tiny") # (down)load dataset

rng = Xoshiro(17)
perm = randperm(rng, length(data))
n_train = round(Int, 0.9 * length(data))
train_raw = data[perm[1:n_train]]
test_raw = data[perm[n_train+1:end]]

println("Silicon dataset: $(length(data)) total structures (train=$(length(train_raw)), test=$(length(test_raw)))")

Eref = [:Si => -158.54496821] # single silicon atom DFT energy
model = ace1_model(; elements=[:Si], order=3, totaldegree=10, rcut=5.5, Eref=Eref) # ACE basis
n_basis = length(model.ps.WB) + length(model.ps.Wpair)

weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0)) # assign more weight to energy rows
datakw = (energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
make_atomsdata(xs) = [ACEpotentials.AtomsData(s; weights=weights, v_ref=model.model.Vref, datakw...) for s in xs]
train = make_atomsdata(train_raw)
test = make_atomsdata(test_raw)

# assemble matrices for linear problem
A_tr, Y_tr, W_tr = ACEfit.assemble(train, model)
A_te, Y_te, W_te = ACEfit.assemble(test, model)

# ACE smoothness prior (feature rescaling/diagonal preconditioner)
P = ACEpotentials.Models.algebraic_smoothness_prior(model.model; p=4)
Ap_tr = Diagonal(W_tr) * (A_tr / P)
Yp_tr = W_tr .* Y_tr

# fit POPS model
pops = fit(POPSModel, Ap_tr, Yp_tr;
    prior_covariance=1e-4,
    leverage_percentile=0.1)

println("fitted POPS ensemble with rank=$(length(pops.lower_bounds)) ($(size(pops.pops_corrections,1)) data points used)")

# apply preconditioner
A_te_p = A_te / P

# inference
pred = predict(pops, A_te_p;
    return_bounds=true, return_std=true, level=1.0,
    min_samples=5000, max_samples=5000,
    sampling_method=:sobol)


# gather results and print

function data_masks(data_list, Ntot)
    e_mask = falses(Ntot)
    f_mask = falses(Ntot)
    v_mask = falses(Ntot)
    row = 0
    for d in data_list
        N_at = length(d.system.atom_data.position)
        nobs = ACEfit.count_observations(d)
        e_mask[row+1] = true
        f_mask[row+2:row+1+3*N_at] .= true
        v_mask[row+2+3*N_at:row+nobs] .= true   # empty for isolated atom

        row += nobs
    end
    return e_mask, f_mask, v_mask
end

e_mask, f_mask, v_mask = data_masks(test, length(Y_te))

residual = pred.mean .- Y_te
is_in_bounds = pred.lower .< Y_te .< pred.upper

channels = (
    E=(label="Energies", mask=e_mask),
    F=(label="Forces", mask=f_mask),
    V=(label="Virials", mask=v_mask),
)

stats = map(channels) do c
    m = c.mask
    (
        label=c.label,
        err=mean(abs, residual[m]),
        pct_inbounds=round(100 * mean(is_in_bounds[m]), digits=2),
        rel_err=mean(abs, pred.std[m] ./ residual[m]),
    )
end

for s in stats
    println("="^10, " $(s.label) ", "="^10)
    println("% in POPS bounds: $(s.pct_inbounds)")
    println("Average error: $(s.err)")
    println("Average σ/error: $(s.rel_err)")
end


