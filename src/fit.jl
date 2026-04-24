# TODO add support for structured weight matrices for linearly related multivariate outputs (e.g. look into Kronecker.jl)

_resolve_prior(::Nothing, P::Int, ::Type{T}) where {T} = zeros(T, P, P) # No regularization
_resolve_prior(alpha::Real, P::Int, ::Type{T}) where {T} = Matrix{T}(T(alpha) * I, P, P)
_resolve_prior(d::AbstractVector, P::Int, ::Type{T}) where {T} = diagm(T.(d))
_resolve_prior(Sigma::AbstractMatrix, ::Int, ::Type{T}) where {T} = Matrix{T}(Sigma)


"""
    StatsAPI.fit(::Type{POPSModel}, X, y; prior_covariance=nothing,
                leverage_percentile=0.5, rank_threshold=nothing)

Fit a POPS hypercube regression model.

# Arguments
- `X::AbstractMatrix`: feature matrix, `N × P`
- `y::AbstractVecOrMat`: response, length-`N` vector or `N × D` matrix

# Keyword arguments
- `prior_covariance`: prior covariance `Σ₀` (P × P). Accepts `nothing` (Σ₀=0),
a scalar (`αI`), a length-`P` vector (diagonal), or a `P × P` matrix.
For multivariate regression, we assume a separable prior `Γ = I_D ⊗ Σ₀`.
- `leverage_percentile`: only points with leverage at or above this quantile
contribute to the hypercube fit (default `0.5`).
- `rank_threshold`: relative singular-value threshold for computing effective rank.
Default `eps(T) * max(M, P·D)`, where M is the number of samples used for fitting
- `fit_intercept` : whether to add a constant colums to the feature matrix (default false)
"""
function StatsAPI.fit(::Type{POPSModel}, X::AbstractMatrix, y::AbstractVecOrMat;
    prior_covariance=nothing,
    leverage_percentile=0.5,
    rank_threshold=nothing,
    fit_intercept=false,
    verbose=false)

    @assert size(X, 1) == size(y, 1) "Number of rows in X and y must match"

    N, P = size(X)
    FT = float(promote_type(eltype(X), eltype(y)))
    lp = FT(leverage_percentile)

    X_ = Matrix{FT}(X)

    is_univariate = ndims(y) == 1

    y_ = is_univariate ? reshape(Vector{FT}(y), :, 1) : Matrix{FT}(y)
    D = size(y_, 2)

    if fit_intercept
        X_ = [ones(FT, N) X_]
        P += 1
    end

    Sigma0 = _resolve_prior(prior_covariance, P, FT)
    C_mat = Symmetric(X_' * X_ + Sigma0 / N) # P × P, regularized covariance matrix
    C_fact = cholesky(C_mat)

    A = C_fact \ X_'                       # P × N, influence matrix
    h = vec(sum(X_ .* A'; dims=2))         # N, leverage scores diag(X C⁻¹ X')

    w = A * y_                             # P × D ridge solution (global loss minimizer)
    residuals = y_ - X_ * w                # N × D

    # keep high-leverage points
    mask = lp > 0 ? (h .>= quantile(h, lp)) : trues(N)
    M = sum(mask)

    A_m = @view A[:, mask]                       # P × M
    residuals_m = @view residuals[mask, :]       # M × D
    h_m = @view h[mask]                          # M

    # compute POPS corrections
    # i-th POPS-constrained minimizer = ridge_solution + (1/leverage[i]) A[i] residual[i]ᵀ  (P × D), stacked as (M, P, D)
    scaled = (A_m ./ h_m')'                # M × P, row i = A[i]ᵀ / leverage[i]
    T_corr = reshape(scaled, M, P, 1) .* reshape(residuals_m, M, 1, D)
    T_corr_mat = reshape(T_corr, M, P * D) # M × (P·D)

    # Hypercube bounds  (SVD + rank truncation)
    rt = isnothing(rank_threshold) ? FT(eps(FT) * max(M, P * D)) : FT(rank_threshold)
    F = svd(T_corr_mat)
    R = max(count(>(rt * F.S[1]), F.S), 1)

    (R > D) && verbose && @warn "effective rank ($R) > output dimension ($D). Predictive entropies will use Gaussian approximation."

    V_R = F.V[:, 1:R]                      # (P·D) × R
    T_tilde = T_corr_mat * V_R             # M × R
    lower = ntuple(r -> minimum(@view T_tilde[:, r]), R)
    upper = ntuple(r -> maximum(@view T_tilde[:, r]), R)

    return POPSModel{R,FT,typeof(prior_covariance)}(;
        prior_covariance,
        leverage_percentile=lp,
        rank_threshold=rt,
        coef=w,
        fit_intercept=fit_intercept,
        is_univariate=is_univariate,
        pops_corrections=T_corr,
        residuals=residuals,
        leverage_scores=h,
        C=C_fact,
        rotation=V_R,
        lower_bounds=lower,
        upper_bounds=upper,
    )
end