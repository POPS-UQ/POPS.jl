```@meta
CurrentModule = POPS
```

# Example: ACE interatomic potential

In this example, we fit a linear ACE (Atomic Cluster Expansion) interatomic potential
on the `Si_tiny` silicon dataset from [Bartok et. al, 2018](https://doi.org/10.1103/PhysRevX.8.041048).
The dataset is provided as an artifact by [`ACEpotentials.jl`](https://github.com/ACEsuit/ACEpotentials.jl).

In this test, we use POPS regression to attach predictive uncertainties to the energies, forces and virials predicted
on a held-out test set.

The full script is available in the repository under
[`examples/ace/pops_ace.jl`](https://github.com/noeblassel/POPS.jl/blob/main/examples/ace/pops_ace.jl), and can be run by following the instructions described [here](#running-the-examples).

## Setting up an ACE linear regression problem

We start by following the `ACEpotentials` workflow, first loading the dataset.
```julia
using ACEpotentials, ACEfit
using POPS
using LinearAlgebra, Random, Statistics, DelimitedFiles

data, _, _ = ACEpotentials.example_dataset("Si_tiny")

rng = Xoshiro(17)
perm = randperm(rng, length(data))
n_train = round(Int, 0.9 * length(data))
train_raw = data[perm[1:n_train]]
test_raw  = data[perm[n_train+1:end]]
```
The next step is to define an ACE linear model, and compute the design matrices from the ACE basis functions.
```
Eref  = [:Si => -158.54496821]
model = ace1_model(; elements=[:Si], order=3, totaldegree=10, rcut=5.5, Eref=Eref)

weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0, "V" => 1.0))
datakw  = (energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
make_atomsdata(xs) =
    [ACEpotentials.AtomsData(s; weights=weights, v_ref=model.model.Vref, datakw...) for s in xs]

train = make_atomsdata(train_raw)
test  = make_atomsdata(test_raw)

A_tr, Y_tr, W_tr = ACEfit.assemble(train, model)
A_te, Y_te, W_te = ACEfit.assemble(test,  model)
```
ACE linear models are usually fitted in a preconditioned basis. Here, we use the so-called algebraic smoothness prior provided by `ACEpotentials.jl`.

```julia
P    = ACEpotentials.Models.algebraic_smoothness_prior(model.model; p=4)
Ap_tr = Diagonal(W_tr) * (A_tr / P)
Yp_tr = W_tr .* Y_tr
```

## Fitting

We now fit the POPS hypercube ensemble. A small regularization term (the prior covariance, which corresponds to the ridge regression parameter) is used,
as well as a  moderate leverage threshold (only the top 90% leverage points are
kept to fit the hypercube ensemble).

```julia
pops = fit(POPSModel, Ap_tr, Yp_tr;
    prior_covariance=1e-4,
    leverage_percentile=0.1)

println("rank=$(length(pops.lower_bounds)), points used=$(size(pops.pops_corrections, 1))")
```

## Inference and uncertainty quantification

We compute mean predictions on the test set, as well as posterior predictive uncertainty measures: min–max bounds and empirical standard deviations.

```julia
A_te_p = A_te / P # preconditioned test features

pred = predict(pops, A_te_p;
    return_bounds=true, return_std=true,
    level=1.0,               
    min_samples=5000,
    sampling_method=:sobol)
```

## Evaluating uncertainty calibration

We evaluate the error calibration of the posterior by comparing true residuals to the predicted uncertainties on the test set. These are broken down by property.

```julia
function data_masks(data_list, Ntot) # computes boolean masks to separate rows corresponding to energies, forces and virials
    e_mask = falses(Ntot); f_mask = falses(Ntot); v_mask = falses(Ntot)
    row = 0
    for d in data_list
        N_at = length(d.system.atom_data.position)
        nobs = ACEfit.count_observations(d)
        e_mask[row+1] = true
        f_mask[row+2:row+1+3*N_at] .= true
        v_mask[row+2+3*N_at:row+nobs] .= true
        row += nobs
    end
    return e_mask, f_mask, v_mask
end

e_mask, f_mask, v_mask = data_masks(test, length(Y_te))

residual = pred.mean .- Y_te
is_in_bounds = pred.lower .< Y_te .< pred.upper

for (label, mask, unit) in (("Energies", e_mask, "eV"), ("Forces", f_mask, "eV/Å"), ("Virials", v_mask, "eV"))
    println("="^10, " $label ", "="^10)
    println("$(sum(is_in_bounds[mask]))/$((sum(mask))) in POPS bound")
    println("Mean residual ($unit) :    ", mean(abs, residual[mask]))
    println("Mean posterior σ/residual :  ", mean(abs, pred.std[mask] ./ residual[mask]))

end
```