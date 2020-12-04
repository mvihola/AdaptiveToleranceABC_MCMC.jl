# AdaptiveToleranceABC_MCMC.jl

This is a Julia package which implements the adaptive tolerance ABC-MCMC with post-correction, as suggested in [Vihola & Franks, 2002](https://doi.org/10.1093/biomet/asz078) (See also the [Preprint](https://arxiv.org/abs/1902.00412)). The package is essentially the same code as [used in the paper](https://bitbucket.org/mvihola/abc-mcmc/src), but rewritten as a stand-alone package, and includes slight code generalisations and refinements.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/mvihola/AdaptiveToleranceABC_MCM.git")
```

## Getting started

```julia
using AdaptiveToleranceABC_MCMC
# Standard normal prior
pr(θ) = -.5θ[]^2 
# Standard normal summaries with mean θ
function sim!(s, θ, rng)
    s[] = θ[] + randn(rng)
    nothing
end
# Run ABC-MCMC
out = abc_mcmc([0.0], pr, [1.0], sim!, 10_000)
# Show post-corrected estimates:
est = abc_postprocess(out, x->x)
plot_abc(est)
```

## With Distributions and LabelledArrays

The following example demonstrates use with [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) and [LabelledArrays.jl](https://github.com/SciML/LabelledArrays.jl): parameters are mean `μ` and log standard deviation `log_σ`, and the summaries are the sample mean and sample variance calculated from four independent N(μ,σ) samples.

```julia
using Distributions, LabelledArrays, AdaptiveToleranceABC_MCMC

# Diffuse prior
function pr(θ)
    logpdf(Normal(0,10), θ.μ) + logpdf(Normal(0,10), θ.log_σ)
end

# Simulation function (using let block to define once-allocated scratch vector x)
sim_normals! = let x = zeros(4)
    function(s, θ, rng)
        σ = exp(θ.log_σ)
        for i = 1:4
            x[i] = rand(rng, Normal(θ.μ, σ))
        end
        s.m = mean(x)
        s.s = var(x)
        nothing
    end
end

# Initial parameter value
θ = LVector(μ=0.0, log_σ=0.0)
# Observed summary
s_obs = LVector(m=0.0, s=0.5^2)
# Run the adaptive tolerance ABC-MCMC
out = abc_mcmc(θ, pr, s_obs, sim_normals!, 200_000)

# Test function:
f(θ) = (θ.μ,exp(θ.log_σ))
# Do post-processing
est = abc_postprocess(out, f)
# Plot the estimates 
plot_abc(est; min_tol=0.02, labels=["μ","σ"])

# (Experimental) regression correction
est_reg = abc_postprocess(out, f, LinRange(0.07, 0.2, 100); regress=true)
plot_abc(est_reg, labels=["μ","σ"])
```