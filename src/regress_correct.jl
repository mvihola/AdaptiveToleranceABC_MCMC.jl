@inline function safe_regress!(coef, XX, X, W, F_)
    try
        coef .= XX\(X'F_)
    catch
        @warn "Regression failed"
        coef .= 0.0
        coef[1] = dot(W, F_)
    end
    nothing
end

@inline function intercept_var_factor(XWX::Matrix{FT}) where {FT <: AbstractFloat}
    try
        max(zero(FT), inv(XWX)[1,1])
    catch
        FT(Inf)
    end
end

"""
    est = abc_regress(out, f, tol; kernel=out.cutoff.fun)

Calculate post-correction estimates for ABC-MCMC output using regression
correction of Beaumont (2002), and producing approximate confidence
intervals similar to `abc_postprocess`

# Arguments
- `out`: Output of the `abc_mcmc` function (with `record_summaries=true`)
- `f`: Function of interest.
- `tol`: Vector of tolerance values for which estimates are calculated.
- `kernel`: Kernel function used in regression correction.
- `est`: Estimates which may be inspected by `ABCPlots.plot_abc(est)`
"""
function abc_regress(out, f::Function, tol::T;
    kernel::Function=out.cutoff.fun, 
    normal_quantile=FT(1.96),
    iact=nothing) where {FT <: AbstractFloat, T <: AbstractVector{FT}}
    if typeof(out.Summary) == Missing
        error("Summary statistics not recorded; please re-run abc_mcmc with 'record_summaries=true'")
    end
    F = eval_testfunction(out, f)
    (d_f,n) = size(F)
    n_tol = length(tol)
    E = Matrix{FT}(undef, d_f, n_tol)
    S = Matrix{FT}(undef, d_f, n_tol)
    ci_L = Matrix{FT}(undef, d_f, n_tol)
    ci_U = Matrix{FT}(undef, d_f, n_tol)

    d_s = size(out.Summary, 1)
    X = Matrix{FT}(undef, n, d_s+1)
    WX = Matrix{FT}(undef, n, d_s+1)
    W = Vector{FT}(undef, n)
    F_ = Vector{FT}(undef, n)
    phi_0 = out.cutoff.(out.Dist)
    invalid = phi_0 .== zero(FT)

    # Construct the regression data matrix X:
    for i in 1:n
        X[i, 1] = one(FT)
        X[i, 2:end] .= out.Summary[:,i] .- out.s_obs
    end

    # Calculate the integrated autocorrelation for
    # regression-corrected samples (using all!)
    tau0 = Vector{FT}(undef, d_f)
    XX = X'*X
    W .= one(FT)/n

    coef = Vector{FT}(undef, d_s+1)

    for j in 1:d_f
        F_ .= F[j,:]
        safe_regress!(coef, XX, X, W, F_)
        F_ .= F[j,:]
        F_ -= X*coef
        tau0[j] = iact(F_)
    end

    for i in 1:n_tol
        tol_ = tol[i]
        # Calculate unnormalised weights
        map!(x -> kernel(x/tol_), W, out.Dist)
        W .= W./phi_0
        W[invalid] .= zero(FT)

        # Normalise (and warn if cannot)
        Ws = sum(W)
        if Ws == zero(FT)
            @warn string("No matching observations for tolerance ", tol_)
            continue
        end
        W /= Ws

        # Calculate weighted regression
        WX .= X
        @inbounds for j in 1:n
            WX[j,:] *= W[j]
        end
        # The 'projection matrix' for tolerance tol_
        XWX = X'*WX
        for j in 1:d_f
            F_ .= F[j,:]
            # Calculate the regression coefficients:
            safe_regress!(coef, XWX, WX, W, F_)
            #coef = XWX\(WX'F_)
            # The mean estimate is simply the intercept:
            alpha = coef[1]
            E[j,i] = alpha
            # F_ = -(regression-corrected values)
            mul!(F_, X, coef)
            F_ -= F[j,:]
            # Calculate S:
            s_ = zero(FT)
            @inbounds for k in 1:n
                s_ += W[k]^2 * F_[k]^2
            end
            s_ *= intercept_var_factor(XWX)
            S[j,i] = s_
            # Confidence bounds:
            dx = normal_quantile*sqrt(s_*tau0[j])
            ci_L[j,i] = alpha - dx
            ci_U[j,i] = alpha + dx
        end
    end
    (E=E, S=S, eps=tol, ci_L=ci_L, ci_U=ci_U, normal_quantile=normal_quantile)
end
