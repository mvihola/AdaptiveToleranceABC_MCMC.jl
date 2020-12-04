"""
    plot_abc(est, [min_tol])

Plot ABC post-correction estimators with related 95% confidence intervals.

NB: "using Plots" must be called before calling this function.

# Arguments
- `est`: Output of `abc_postprocess`.
- `min_tol`: Minimum tolerance (default 0.0).
- `show`: Whether the plot is shown
"""
function plot_abc(est; min_tol=0.0, gather=true, labels=nothing,
    xlabel="ϵ")
    lims(x) = (minimum(x), maximum(x))
    d, n = size(est.E)
    v = est.eps .>= min_tol
    eps_ = est.eps[v]
    if length(eps_) == 0
        warning("All distances less than min_tol")
        return nothing
    end
    p = Vector{Plots.Plot}(undef, 0)
    for i in 1:d
        E_ = est.E[i,v]; L_ = est.ci_L[i,v];  U_ = est.ci_U[i,v]
        push!(p, Plots.plot(eps_, E_, ribbon=[E_-L_, U_-E_], color=:black,
                          fillalpha=0.3, legend=false, xlim=lims(eps_),
                          framestyle=:axes))
        if !(labels isa Nothing)
            Plots.plot!(p[end], ylabel=labels[i])
        end
    end
    if gather
        Plots.plot!(p[end], xlabel=xlabel)
        Plots.plot(p..., layout=(d,1))
    else
        p
    end
end
