# TODO for multivariate outputs, stds/bounds should really be covariance matrices and trust regions, respectively. Need to represent these trust regions.


function _clipped_bounds(lower::NTuple{R,T}, upper::NTuple{R,T}, percentile::Real) where {R,T}
    alpha = percentile^inv(R) # keep proportion percentile of hypercube mass
    margin = T((1 - alpha) / 2)
    widths = upper .- lower
    l = lower .+ margin .* widths
    u = upper .- margin .* widths
    return l, u
end

_hypercube_vol(lb, ub) = prod(u - l for (l, u) = zip(lb, ub))


"""
    sample([rng,] model::POPSModel, n::Int; percentile=1.0, sampling_method=:uniform) -> Array

Generate `n` parameter samples from the POPS-constrained hypercube.

Returns a `P × D × n` array where each slice `samples[:, :, i]` is a full
coefficient matrix (ridge solution + perturbation), directly usable as
`X * samples[:, :, i]` to obtain predictions.

# Keyword arguments
- `percentile_clipping`: inference-time clipping parameter (default `1.0`, full hypercube).
E.g. `percentile=0.95` shrinks the hypercube uniformly to 95% of its R-dimensional volume, where R is the effective rank
- `sampling_method`: `:uniform` (default, PRNG) or `:sobol` (low-discrepancy quasi-Monte Carlo). The `rng` argument is ignored when `:sobol` is used.
"""
function sample(rng::AbstractRNG, m::POPSModel{R,T}, n::Int;
    percentile::Real=1.0,
    sampling_method::Symbol=:uniform) where {R,T}

    l, u = _clipped_bounds(m.lower_bounds, m.upper_bounds, percentile)
    P, D = size(m.weights)
    widths = u .- l
    samples = Array{T,3}(undef, P, D, n)

    if sampling_method == :uniform
        for i in 1:n
            x = l .+ widths .* rand(rng, NTuple{R,T})
            dW = reshape(m.rotation * collect(x), P, D)
            samples[:, :, i] = m.weights .+ dW
        end
    elseif sampling_method == :sobol
        s = SobolSeq(collect(l), collect(u))
        for i in 1:n
            x = next!(s)
            dW = reshape(m.rotation * x, P, D)
            samples[:, :, i] = m.weights .+ dW
        end
    else
        throw(ArgumentError("unknown sampling_method $(sampling_method); expected :uniform or :sobol"))
    end

    return m.is_univariate ? dropdims(samples; dims=2) : samples
end

sample(m::POPSModel, n::Int; percentile::Real=1.0, sampling_method::Symbol=:uniform) =
    sample(Random.default_rng(), m, n; percentile, sampling_method)

"""
    StatsAPI.predict(m::POPSModel, X::AbstractMatrix; return_bounds=true, return_std=false,
                    return_entropy=false, return_samples=false, 
                    num_samples=max(2, 10 * ceil(Int, _pops_volume(m))), level=0.95,
                    percentile=1.0, rng=Random.default_rng())

Computes predictions for test feature matrix `X`, with uncertainty quantification 
from the POPSModel posterior ensemble.

# Arguments
- `m::POPSModel`: A POPS model with fitted hypercube bounds.
- `X`: The feature matrix for points at which inference is performed. (size n × P for a batch of n points)
A constant feature column is added if `m.fit_intercept` is `true`.

# Keyword arguments
- `return_bounds=true` : return empirical bounds (controlled by `level`) from the predictive distribution at each input.
- `return_std=false` : return empirical stds from predictive distribution at each input.
- `return_entropy=false` : return the (analytical) differential entropies of the predictive distribution at each input (restricted to its support if effective rank is less than the output dimension).
- `return_samples=false` : return the empirical samples used for prediction at each input.
- `sampling_density` : governs the number of samples used for uncertainty quantification. The number of samples is `sampling_density` times the hypervolume of the posterior support (clipped using the `percentile` parameter)
- `level=0.95` : determines the quantiles for the empirical bounds (e.g., `0.95` gives the 2.5% and 97.5% quantiles). Set to `1.0` for min-max bounds.
- `percentile=1.0` : fraction of hypercube volume (centered around mean prediction) to draw samples from.
- `min_samples=30` : minimum number of posterior samples (useful for well-specified models)
- `max_samples=1000` : maximum number of posterior samples (useful for poorly-specified or high-dimensional models)
- `rng` : optional PRNG
- `sampling_method=:uniform`. The method used to generate samples, currently implemented methods are `:uniform` and `:sobol` (low-discrepancy quasi Monte Carlo sampling)

# Returns
- A `NamedTuple` with the field `mean`, and any of the fields: `lower` and `upper` (if `return_bounds` is `true`), `std`, `entropy`, and `samples`.

Output quantities are naturally squeezed depending on the value of `m.is_univariate`, i.e., the predicted quantities will be one-dimensional vectors if this value is `true`.
"""
function StatsAPI.predict(m::POPSModel{R,T}, X::AbstractMatrix;
    return_bounds::Bool=true,
    return_std::Bool=false,
    return_entropy::Bool=false,
    return_samples::Bool=false,
    sampling_density::Real=10.0,
    sampling_method::Symbol=:uniform,
    level::Real=0.95,
    percentile::Real=1.0, # TODO remove redundant parameters
    min_samples::Int=30,
    max_samples::Int=1000,
    rng::AbstractRNG=Random.default_rng()) where {R,T}

    X_ = m.fit_intercept ? hcat(ones(T, size(X, 1)), T.(X)) : T.(X) # (N_test × P)
    N_test = size(X_, 1)
    P, D = size(m.weights)

    _squeeze(y) = m.is_univariate ? dropdims(y, dims=2) : y

    l_tup, u_tup = _clipped_bounds(m.lower_bounds, m.upper_bounds, percentile)
    posterior_vol = _hypercube_vol(l_tup, u_tup)
    num_samples = clamp(ceil(Int, posterior_vol * sampling_density), min_samples:max_samples)


    W_samples = sample(rng, m, num_samples; percentile, sampling_method) # (P × D × num_samples)
    Y_samples = reshape(X_ * reshape(W_samples, P, D * num_samples), N_test, D, num_samples) # (N_test × D × num_samples)

    mean_pred = dropdims(mean(Y_samples, dims=3), dims=3) # (N_test × D)

    result = (; mean=_squeeze(mean_pred))

    if return_bounds
        p_lo = T((1 - level) / 2)
        p_hi = T((1 + level) / 2)

        lower = Matrix{T}(undef, N_test, D)
        upper = Matrix{T}(undef, N_test, D)

        for j in 1:D, i in 1:N_test
            lo, hi = quantile(view(Y_samples, i, j, :), (p_lo, p_hi))
            lower[i, j] = lo
            upper[i, j] = hi
        end

        result = merge(result, (; lower=_squeeze(lower), upper=_squeeze(upper)))
    end


    if return_std
        std_pred = dropdims(std(Y_samples; dims=3), dims=3)
        result = merge(result, (; std=_squeeze(std_pred)))
    end

    if return_entropy

        widths = collect(u_tup) .- collect(l_tup)
        sum_log_w = sum(log, widths)

        # G_tensor maps hypercube u ∈ R^R to predictions y ∈ R^D for each point x

        G_tensor = Array{T,3}(undef, N_test, D, R)
        for r in 1:R
            G_tensor[:, :, r] = X_ * reshape(m.rotation[:, r], P, D)
        end

        H_array = Array{T,1}(undef, N_test)

        for i in 1:N_test
            G_i = view(G_tensor, i, :, :) # Jacobian of model pushforward θ → (θ⊤)x_i (D × R)

            M = R <= D ? Symmetric(G_i' * G_i) : Symmetric(G_i * G_i') # change Gram matrix depending on support dimensionality ?

            log_det_M = logdet(M)

            if log_det_M == -Inf
                @warn "Singular predictive distribution at data point $i"
                H_array[i] = -Inf
            else
                H_array[i] = sum_log_w + T(0.5) * logdet(M)
            end

        end

        result = merge(result, (; entropy=H_array))
    end

    if return_samples
        result = merge(result, (; samples=Y_samples))
    end

    return result
end