# POPS.jl

A library of probabilistic surrogate models, targeting the low-noise, mispecified regime.

The package takes its name from the POPS regression algorithm ([Perez & Swinburne, 2025](https://doi.org/10.1088/2632-2153/ad9fce)). 
Implemented models follow the [StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl) specification.

## Implemented features
- Multivariate POPS regression
- Parameter uncertainty quantification for univariate and multivariate predictive posteriors: confidence intervals, min-max bounds and entropy estimates