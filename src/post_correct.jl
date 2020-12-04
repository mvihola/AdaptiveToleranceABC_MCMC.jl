function eval_testfunction(out, f)
    theta = deepcopy(out.theta0)
    x = f(theta)
    d = length(x); n = size(out.Theta)[2]
    F = zeros(eltype(x), d, n)
    for i = 1:n
        theta .= out.Theta[:, i]
        F[:,i] .= f(theta)
    end
    F
end


# When correction kernel is simple, we may calculate estimates for all epsilon
# (and then find the suitable epsilon among)
function is_est(X::Matrix{FT}, D::Vector{FT}, phi::ABCCutoff,
    kernel::ABCSimpleCutoff, eps; normal_quantile=FT(1.96), 
    iact=nothing) where {FT <: AbstractFloat}

    eps0 = phi.tol

    d, n = size(X)
    I = sortperm(D); D_ = D[I]
    if typeof(eps) == Nothing
        n_eps = n
        eps = D_
    else
        n_eps = length(eps)
    end

    E = Matrix{FT}(undef,d,n)
    S = Matrix{FT}(undef,d,n)
    E_eps = Matrix{FT}(undef,d,n_eps)
    S_eps = Matrix{FT}(undef,d,n_eps)
    ci_L = Matrix{FT}(undef,d,n_eps)
    ci_U = Matrix{FT}(undef,d,n_eps)

    I = sortperm(D); D_ = D[I]
    phi_0 = phi.(D_)
    invalid = phi_0 .== zero(FT)
    xi_1 = one(FT)./phi_0
    xi_1[invalid] .= zero(FT)
    xi_1 /= sum(xi_1)
    for i in 1:d
        X_ = X[i,I]
        xi_f = X_.*xi_1
        E[i,:] = cumsum(xi_f)./cumsum(xi_1)
        S[i,:] = (cumsum(xi_f.^2) - 2E[i,:].*cumsum(xi_f.*xi_1)
                  + cumsum(xi_1.^2).*E[i,:].^2) ./ cumsum(xi_1).^2
    end
    if typeof(eps) == Nothing
        E_eps = E; S_eps = S
    else 
        j = 1
        for k in 1:n_eps
            while j<n && D_[j+1] < eps[k]
                j += 1
            end
            E_eps[:,k] = E[:,j]
            S_eps[:,k] = S[:,j]
        end
    end
    for i in 1:d
        dx = normal_quantile*sqrt.(max.(S_eps[i,:],0)*iact(X[i,:]))
        ci_L[i,:] = E_eps[i,:] - dx; ci_U[i,:] = E_eps[i,:] + dx
    end
    (E=E, S=S, eps=eps, ci_L=ci_L, ci_U=ci_U, normal_quantile=normal_quantile)
end

# For general correction, calculate directly from the definition
function is_est(X::Matrix{FT}, D::Vector{FT}, phi::ABCCutoff,
    kernel::ABCGenCutoff, eps; normal_quantile=FT(1.96), 
    iact=nothing) where {FT <: AbstractFloat}
    
    eps isa Nothing || error("Must supply ϵ with general cutoff")
    d, n = size(X)
    n_eps = length(eps)
    E = Matrix{FT}(undef, d, n_eps)
    S = Matrix{FT}(undef, d, n_eps)
    ci_L = Matrix{FT}(undef, d, n_eps)
    ci_U = Matrix{FT}(undef, d, n_eps)
    # Evaluate the cutoff for eps0, and the positive indices
    phi_0 = phi.(D)
    invalid = phi_0 .== zero(FT)
    for k in 1:n_eps
        U = kernel.fun.(D/eps[k]) ./ phi_0
        U[invalid] .= zero(FT)
        W = U/sum(U)
        for i in 1:d
            E[i,k] = sum(W .* X[i,:])
            S[i,k] = sum(W.^2 .* (X[i,:] .- E[i,k]).^2)
        end
    end
    for i in 1:d
        dx = normal_quantile*sqrt.(max.(S[i,:],0)*iact(X[i,:]))
        ci_L[i,:] = E[i,:] - dx
        ci_U[i,:] = E[i,:] + dx
    end
    (E=E, S=S, eps=eps, ci_L=ci_L, ci_U=ci_U, normal_quantile=normal_quantile)
end

"""
    est = abc_postprocess(out, f, [tol]; regress=false, kernel=out.cutoff,
                          normal_quantile=1.96)

Calculate post-correction estimates for ABC-MCMC output.

# Arguments
- `out`: Output of the `abc_mcmc` function.
- `f`: Function of interest.
- `tol`: Vector of tolerance values for which estimates are calculated.
         If not given (or `nothing`), and `out.cutoff` is simple, then
         correction calculated to all tolerances.
- `regress`: Whether regression correction should be used.
- `kernel`: Kernel for post-correction. (NB kernel.tol is ignored!)
- `normal_quantile`: The desired normal quantile (default 1.96 -> 95% coverage;
                     any other quantile may be calculated using Distributions as:
                     `-quantile(Normal(), (1.0-coverage)/2)`
- `est`: Estimates
"""
function abc_postprocess(out, f::Function, tol=nothing;
    regress=false, kernel=out.cutoff, normal_quantile=1.96, 
    iact = x -> estimateBM(x)/var(x))
    if regress
        if out.Summary isa Missing
            error("Summary statistics not recorded; please re-run abc_mcmc with 'record_summaries=true'")
        end
        if tol isa Nothing
            error("Must supply tolerances with regression correction")
        end
        return abc_regress(out, f, tol; kernel=kernel.fun,
                                        normal_quantile=normal_quantile, 
                                        iact=iact)
    else
        #F = mapslices(f, out.Theta, dims=1)
        F = eval_testfunction(out, f)
        return is_est(F, out.Dist, out.cutoff, kernel, tol;
                      normal_quantile=normal_quantile, iact=iact)
    end
end
