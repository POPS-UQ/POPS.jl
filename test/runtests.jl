using Test
using POPS
using LinearAlgebra
using Random
using Statistics
using CondaPkg
using PythonCall

@testset "POPS.jl" begin

    rng = Xoshiro(123)

    @testset "basic functionality" begin
        N, P = 100, 5
        X = randn(rng, N, P)
        w_true = randn(rng, P)
        y = X * w_true + 0.1 * randn(rng, N)

        m = fit(POPSModel, X, y)

        @test size(coef(m)) == (P, 1)
        @test nobs(m) == N
        @test dof(m) == P
        @test dof_residual(m) == N - P
        @test size(residuals(m)) == (N, 1)
        @test length(leverage(m)) == N
        @test islinear(m)
        @test isfitted(m)
    end

    @testset "regression" begin
        N, P = 500, 3
        X = randn(rng, N, P)
        w_true = [1.0, -2.0, 0.5]
        y = X * w_true + 0.01 * randn(rng, N)

        m = fit(POPSModel, X, y)
        @test vec(coef(m)) ≈ w_true atol = 0.05
    end

    @testset "prediction" begin
        N, P = 50, 3
        X = randn(rng, N, P)
        y = X * ones(P)

        m = fit(POPSModel, X, y)

        newX = randn(rng, 10, P)
        pred = predict(m, newX)
        @test pred isa NamedTuple
        @test haskey(pred, :mean)
        @test length(pred.mean) == 10
        @test pred.mean ≈ newX * vec(coef(m)) atol = 0.5
    end

    @testset "uq" begin
        N, P = 100, 3
        X = randn(rng, N, P)
        y = X * ones(P) + 0.1 * randn(rng, N)

        m = fit(POPSModel, X, y)
        newX = randn(rng, 5, P)

        pred = predict(m, newX; return_bounds=true, return_std=true,
            return_entropy=true, rng=rng)
        @test haskey(pred, :mean)
        @test haskey(pred, :lower)
        @test haskey(pred, :upper)
        @test haskey(pred, :std)
        @test haskey(pred, :entropy)
        @test all(pred.lower .<= pred.mean .<= pred.upper)
        @test all(pred.std .>= 0)
        @test length(pred.entropy) == 5
    end

    @testset "sampling posterior" begin
        N, P = 200, 4
        X = randn(rng, N, P)
        w_true = randn(rng, P)
        y = X * w_true + 0.1 * randn(rng, N)

        m = fit(POPSModel, X, y)
        S = sample(rng, m, 1000)

        @test size(S, 1) == P
        @test size(S, 2) == 1000

        sample_mean = vec(mean(S; dims=2))
        @test all(isapprox.(sample_mean, vec(coef(m)); atol=1.0))
    end

    @testset "percentile clipping" begin
        N, P = 100, 3
        X = randn(rng, N, P)
        y = X * ones(P) + 0.1 * randn(rng, N)

        m = fit(POPSModel, X, y)

        S_full = sample(rng, m, 500; percentile=1.0)
        S_clip = sample(rng, m, 500; percentile=0.5)

        # Clipped samples should have smaller spread
        spread_full = maximum(S_full; dims=2) - minimum(S_full; dims=2)
        spread_clip = maximum(S_clip; dims=2) - minimum(S_clip; dims=2)
        @test all(spread_clip .<= spread_full .+ eps())
    end

    @testset "priors" begin
        N, P = 50, 3
        X = randn(rng, N, P)
        y = randn(rng, N)

        m0 = fit(POPSModel, X, y) # no regularization
        @test isfitted(m0)

        m1 = fit(POPSModel, X, y; prior_covariance=1.0) # scalar multiple of I
        @test isfitted(m1)

        m2 = fit(POPSModel, X, y; prior_covariance=fill(0.5, P)) # diagonal prior
        @test isfitted(m2)

        A = randn(rng, P, P)
        Σ = 0.1 * (A * A') / P
        m3 = fit(POPSModel, X, y; prior_covariance=Σ) # matrix prior
        @test isfitted(m3)

        m4 = fit(POPSModel, X, y; prior_covariance=Matrix(0.5I, P, P)) # consistency check
        @test coef(m2) ≈ coef(m4)
    end

    @testset "leverage filtering" begin
        N, P = 100, 3
        X = randn(rng, N, P)
        y = randn(rng, N)

        m_full = fit(POPSModel, X, y; leverage_percentile=0.0)
        m_filt = fit(POPSModel, X, y; leverage_percentile=0.95)

        @test size(m_filt.pops_corrections, 1) <= size(m_full.pops_corrections, 1)
        @test coef(m_full) ≈ coef(m_filt)
    end

    @testset "pops minimizers interpolate (univariate)" begin
        N, P = 100, 5
        X = randn(rng, N, P)
        w_true = randn(rng, P)
        y = X * w_true + 0.5 * randn(rng, N)

        m = fit(POPSModel, X, y; leverage_percentile=0.0)
        w = coef(m)                          # P × 1
        T_corr = m.pops_corrections          # N × P × 1

        for i in 1:N
            w_corrected = w .+ T_corr[i, :, :]  # P × 1
            @test X[i, :]' * w_corrected ≈ [y[i]] atol = 1e-10
        end
    end

    @testset "pops minimizers interpolate (multivariate)" begin
        N, P, D = 80, 4, 3
        X = randn(rng, N, P)
        W_true = randn(rng, P, D)
        Y = X * W_true + 0.5 * randn(rng, N, D)

        m = fit(POPSModel, X, Y; leverage_percentile=0.0)
        w = coef(m)                          # P × D
        T_corr = m.pops_corrections          # N × P × D

        for i in 1:N
            w_corrected = w .+ T_corr[i, :, :]  # P × D
            @test X[i, :]' * w_corrected ≈ Y[i, :]' atol = 1e-10
        end
    end


    @testset "multivariate regression" begin
        N, P, D = 100, 3, 2
        X = randn(rng, N, P)
        W_true = randn(rng, P, D)
        Y = X * W_true + 0.1 * randn(rng, N, D)

        m = fit(POPSModel, X, Y)

        @test size(coef(m)) == (P, D)
        @test size(residuals(m)) == (N, D)
        @test nobs(m) == N

        S = sample(rng, m, 50)
        @test size(S) == (P, D, 50)

        pred = predict(m, X; rng=rng)
        @test size(pred.mean) == (N, D)
    end

    @testset "fit_intercept" begin
        N, P = 100, 3
        X = randn(rng, N, P)
        w_true = [1.0, -2.0, 0.5]
        b_true = 3.0
        y = X * w_true .+ b_true + 0.01 * randn(rng, N)

        m = fit(POPSModel, X, y; fit_intercept=true)

        @test size(coef(m), 1) == P + 1
        @test vec(coef(m)) ≈ [b_true; w_true] atol = 0.1
    end

    @testset "row weights" begin
        N, P = 100, 4
        X = randn(rng, N, P)
        w_true = randn(rng, P)
        y = X * w_true + 0.1 * randn(rng, N)

        m_uw = fit(POPSModel, X, y)
        m_w1 = fit(POPSModel, X, y; weights=fill(0.5, N))
        @test coef(m_uw) ≈ coef(m_w1)

        wts = abs.(randn(rng, N)) .+ 0.1
        m_w = fit(POPSModel, X, y; weights=wts)

        sqW = Diagonal(sqrt.(wts))
        w_wls = (sqW * X) \ (sqW * y)
        @test vec(coef(m_w)) ≈ w_wls atol = 1e-10

        m_full = fit(POPSModel, X, y; weights=wts, leverage_percentile=0.0)
        w_fit = coef(m_full)
        T_corr = m_full.pops_corrections
        for i in 1:N
            w_corrected = w_fit .+ T_corr[i, :, :]
            @test X[i, :]' * w_corrected ≈ [y[i]] atol = 1e-10
        end
    end

    @testset "compare against python" begin
        POPSRegression = pyimport("POPSRegression").POPSRegression
        np = pyimport("numpy")

        function pops_fit(X::Matrix{Float64}, y::Vector{Float64};
            resampling_method="sobol",
            posterior="hypercube",
            resample_density=10.0,
            fit_intercept=false,
            leverage_percentile=50.0,
            kwargs...)
            model = POPSRegression(;
                resampling_method, posterior, resample_density,
                fit_intercept, leverage_percentile, kwargs...)
            model.fit(np.array(X), np.array(y))
            model
        end

        function pops_predict(model, X::Matrix{Float64})
            res = model.predict(np.array(X),
                return_std=true, return_bounds=true, return_epistemic_std=true)

            T = Vector{Float64}
            y_mean, y_std, y_max, y_min, y_estd = map(x -> pyconvert(T, x), res)

            return (; y_mean, y_std, y_max, y_min, y_estd)
        end

        # 1D example taken from https://github.com/tomswinburne/popsregression/blob/main/SimpleExample.ipynb

        f(x) = (x^3 + 0.01 * x^4) * 0.1 + 10.0 * sin(x) * x
        N, Nt = 500, 20

        Z = 2.0 .* (1.0 .- rand(rng, N))
        Zt = 2.0 .* (1.0 .- rand(rng, Nt))

        X = [ones(N) Z Z .^ 2 Z .^ 3 Z .^ 4]
        Xt = [ones(Nt) Zt Zt .^ 2 Zt .^ 3 Zt .^ 4]

        Y = f.(Z)

        m_py = pops_fit(X, Y; leverage_percentile=50.0, resampling_method="sobol")
        m_jl = fit(POPSModel, X, Y; leverage_percentile=0.5)

        py_pred = pops_predict(m_py, Xt)


        jl_pred = predict(m_jl, Xt; return_bounds=true, return_std=true,
            sampling_method=:sobol) # use Sobol sequence

        py_coef = pyconvert(Vector{Float64}, m_py.coef_)
        jl_coef = vec(coef(m_jl))

        @info "python coef" py_coef
        @info "julia coef" jl_coef
        @info "mean abs diff (mean)" mean(abs.(py_pred.y_mean .- jl_pred.mean))
        @info "mean abs diff (std)" mean(abs.(py_pred.y_std .- jl_pred.std))

        @test isapprox(py_coef, jl_coef; rtol=0.1)


        @test isapprox(py_pred.y_mean, jl_pred.mean; rtol=0.01)
        @test isapprox(py_pred.y_std, jl_pred.std; rtol=0.1)
        @test isapprox(py_pred.y_max, jl_pred.upper; rtol=0.1)
        @test isapprox(py_pred.y_min, jl_pred.lower; rtol=0.1)


        @info "mean relative difference (mean prediction)" mean(@. abs(py_pred.y_mean - jl_pred.mean) / max.(abs(py_pred.y_mean), abs(jl_pred.mean)))
        @info "mean relative difference (std)" mean(@. abs(py_pred.y_std - jl_pred.std) / max.(abs(py_pred.y_std), abs(jl_pred.std)))
        @info "mean relative difference (max bound)" mean(@. abs(py_pred.y_max - jl_pred.upper) / max.(abs(py_pred.y_max), abs(jl_pred.upper)))
        @info "mean relative difference (min bound)" mean(@. abs(py_pred.y_min - jl_pred.lower) / max.(abs(py_pred.y_min), abs(jl_pred.lower)))
    end


end
