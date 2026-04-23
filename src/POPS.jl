module POPS

import StatsAPI
import StatsAPI: fit, predict, coef, nobs, dof, islinear, isfitted,
    rss, vcov, residuals, leverage, dof_residual

using LinearAlgebra, Statistics
using Random
using Sobol

export POPSModel, sample
export fit, predict, coef, nobs, dof, islinear, isfitted,
    rss, vcov, residuals, leverage, dof_residual

# Helper methods


"""
    POPSModel{R,T,S} <: StatsAPI.RegressionModel

A fitted POPS hypercube regression model.

Implements the POPS-constrained hypercube ansatz from Swinburne & Perez (2025),
providing misspecification-aware parameter uncertainties for linear surrogate models
in the low-noise regime. Supports univariate (`D=1`) and multivariate (`D>1`)
responses under a separable prior `Γ = I_D ⊗ Σ₀`.

# Type parameters
- `R`: effective rank of the POPS correction matrix (dimensionality of the hypercube)
- `T`: element type (e.g. `Float64`)
- `S`: type of the prior covariance specification

# Fitting hyperparameters
- `prior_covariance`: prior covariance specification used during fitting
- `leverage_percentile`: training points with leverage below this quantile are dropped
- `rank_threshold`: relative threshold for determining the effective rank R

# Fit results
- `weights`: ridge solution, `P × D` matrix
- `pops_corrections`: per-point corrections, `M × P × D` tensor (`M` = retained points)
- `residuals`: `N × D` matrix of training residuals
- `leverage_scores`: length-`N` vector
- `C`: Cholesky factor of the regularized Gram matrix `X'X + Σ₀/N`

# Ensemble
- `rotation`: `(P·D) × R` orthonormal basis spanning the hypercube directions in vec-space
- `lower_bounds`, `upper_bounds`: `R`-tuples defining the hypercube extents
"""
@kwdef struct POPSModel{R,T<:AbstractFloat,S} <: StatsAPI.RegressionModel
    # Hyperparameters
    prior_covariance::S
    leverage_percentile::T
    rank_threshold::T
    fit_intercept::Bool
    is_univariate::Bool

    # Fit results
    weights::Matrix{T}
    pops_corrections::Array{T,3}
    residuals::Matrix{T}
    leverage_scores::Vector{T}
    C::Cholesky{T,Matrix{T}}

    # Ensemble
    rotation::Matrix{T}
    lower_bounds::NTuple{R,T}
    upper_bounds::NTuple{R,T}
end


# StatsAPI: StatisticalModel methods

StatsAPI.coef(m::POPSModel) = m.weights
StatsAPI.nobs(m::POPSModel) = size(m.residuals, 1)
StatsAPI.dof(m::POPSModel) = size(m.weights, 1)
StatsAPI.islinear(::POPSModel) = true
StatsAPI.isfitted(::POPSModel) = true
StatsAPI.rss(m::POPSModel) = sum(abs2, m.residuals)
StatsAPI.vcov(m::POPSModel) = inv(m.C)

# StatsAPI: RegressionModel methods

StatsAPI.residuals(m::POPSModel) = m.residuals
StatsAPI.leverage(m::POPSModel) = m.leverage_scores
StatsAPI.dof_residual(m::POPSModel) = StatsAPI.nobs(m) - StatsAPI.dof(m)


include("fit.jl")
include("predict.jl")

end