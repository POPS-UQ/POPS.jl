# POPS.jl

A library of probabilistic surrogate models targeting the low-noise, misspecified regime, written in Julia.
The name comes from the POPS regression algorithm from [Perez & Swinburne (2025)](https://doi.org/10.1088/2632-2153/ad9fce).
                                                                                                                                                                                        
  ## Features                                                                                                                                                                              
                                                                                                                                                                                        
  - Univariate and multivariate POPS hypercube regression for linear models
  - Flexible regularization priors
  - Optional intercept fitting                                                                                                                                                          
  - Leverage-based filtering of training points 
  - Effective-rank detection for posterior parameter distribution                                                                                                                      
  - Posterior sampling with uniform PRNG or Sobol low-discrepancy sequences                                                                                                              
  - Percentile-based clipping of the posterior distribution                                                                                                                                  
  - Predictive uncertainty quantification: min–max bounds, standard deviations, and differential entropy estimates                                                
  - Compliance with the StatsAPI.jl interface                                                                                                                                            
  - Unit testing, including sanity checks against the scikit-learn based [Python implementation](https://github.com/tomswinburne/popsregression)