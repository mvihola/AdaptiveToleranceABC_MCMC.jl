function default_min_tol(est; n_min=20)
    n_min = min(n_min, length(est.eps))
    est.eps[n_min]
end

"""
    plot_abc(est; min_tol=default, gather=true, labels=nothing, xlabel="\\epsilon")

Plot ABC post-correction estimators with related 95% confidence intervals.

# Arguments
- `est`: Output of `abc_postprocess`.
- `min_tol`: Minimum tolerance. By default, 20th smallest sample tolerance.
- `gather`: Whether the individual plots are gathered to one.
- `labels`: The labels of the estimates.
- `xlabel`: The label of the x-axis (tolerance).
"""
function plot_abc(est; min_tol=default_min_tol(est), gather=true, labels=nothing,
    xlabel="\\epsilon")
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
        push!(p, Plots.plot(eps_, E_, ribbon=(E_-L_, U_-E_), color=:black,
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
