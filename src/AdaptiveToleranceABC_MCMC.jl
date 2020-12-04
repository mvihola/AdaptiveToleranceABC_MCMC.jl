
module AdaptiveToleranceABC_MCMC
# Adaptive tolerance ABC-MCMC algorithm with post-correction

export ABCCutoff, ABCSimpleCutoff, ABCGenCutoff, ABCGaussCutoff, ABCEpaCutoff, 
       abc_mcmc, abc_postprocess, ABCdefaultDist, plot_abc

using Statistics, LinearAlgebra, Random, AdaptiveMCMC, 
      MonteCarloMarkovKernels, Plots

# Abstract 'cutoff function' type
abstract type ABCCutoff end

# Simple cutoff at 1.0
mutable struct ABCSimpleCutoff{F,T} <: ABCCutoff where {F<:Function,T}
    tol::T
    fun::F
end

# General cutoff function
mutable struct ABCGenCutoff{F,T} <: ABCCutoff where {F<:Function,T}
    tol::T
    fun::F
end

# Outer constructors for the abstract type:
# if no function supplied, it's simple...
function ABCCutoff(tol::FT, fun=nothing) where {FT <: AbstractFloat}
    if fun isa Nothing
        return ABCSimpleCutoff(tol, x -> (x <= one(FT)))
    else
        return  ABCGenCutoff(tol, fun)
    end
end
# Gaussian
function ABCGaussCutoff(tol::FT) where {FT <: AbstractFloat}
    ABCGenCutoff(tol, x -> exp(-FT(0.5) * x^2))
end
# Epanechnikov
function ABCEpaCutoff(tol::FT) where {FT <: AbstractFloat}
    ABCGenCutoff(tol, x -> x >= one(FT) ? zero(FT) : one(FT) - x^2)
end

# Convenience functions
@inline function (c::ABCGenCutoff)(x)
    c.fun(x/c.tol)
end
@inline function (c::ABCSimpleCutoff)(x)
    c.fun(x/c.tol)
end

@inline function ABCdefaultDist(x::T, y::T) where {
    FT <: AbstractFloat, T <: AbstractVector{FT}}

    s = zero(FT)
    for i in 1:length(x)
        s += (x[i] - y[i])^2
    end
    isnan(s) ? FT(Inf) : sqrt(s)
end

include("abc_mcmc.jl")
include("post_correct.jl")
include("regress_correct.jl")
include("plot_abc.jl")

end # module
