"""
    out = abc_mcmc(theta0, prior, s_obs, sim!, n; [tol=0.0, b=0.25n, adapt_cov=true,
             adapt_tol=true, cutoff=nothing, sc=1.0, tol_targetAcceptance=0.10,
             rng=MersenneTwister(Random.make.seed())], dist=ABCdefaultDist,
             record_summaries=true)

Sample from ABC posterior using tolerance adaptive ABC-MCMC.

# Arguments
- `theta_0`: Initial value of parameter vector
- `prior`: Function returning prior log density values.
- `s_obs`: Observed summary statistics vector
- `sim!`: Function simulating summaries; sim!(s, θ, rng) simulates observed
          summaries corresponding to parameters θ and records them to s.
- `n`: Length of MCMC chain.
- `tol`: Tolerance (initial tolerance if `adapt_tol=true`).
- `b`: Burn-in length. The total number of simulations is b+n.
- `adapt_cov`: Whether covariance adaptation is done.
- `adapt_tol`: Whether tolerance adaptation is done.
- `cutoff`: Cut-off function ϕ. If `nothing`, simple cut-off is used,
            "gaussian" => Gaussian, "epa" => Epanechnikov. For other cut-offs, specifies
            the desired function.
- `sc`: Scaling of proposal distribution (only used if `adapt_cov=false`)
- `tol_targetAcceptance`: Target acceptance rate of tolerance adaptation.
- `rng`: Random number generator
- `dist`: Distance function (default Euclidean distance).
- `record_summaries`: Whether the summaries should be recorded.
- `out`: Output named tuple with fields `Theta` (matrix of d×n, each column
         is simulated parameter), `Dist` (vector of corresponding tolerances),
         `Summary` (matrix of corresponding summaries),
         `cutoff` (the cut-off function w/ tolerance), `Stats` (various
         statistics)
"""
function abc_mcmc(theta0::thetaT, prior::PriT, s_obs::summaryT,
    sim!::SimT, n; tol=zero(FT), 
    b=convert(Int64, ceil(0.25n)),
    adapt_cov::Bool=true, adapt_tol::Bool=true,
    cutoff::CutoffT=nothing, sc=one(FT), stepSize_eta=FT(0.66),
    tol_targetAcceptance=FT(0.10), rng=Random.GLOBAL_RNG,
    record_summaries::Bool=true, dist=ABCdefaultDist) where {
        FT <: AbstractFloat, 
        thetaT <: AbstractVector{FT},
        summaryT <: AbstractVector{FT},
        PriT <: Function, SimT <: Function,
        CutoffT <: Union{Nothing,String,Function,ABCCutoff}}

    d = length(theta0)
    rwm = AdaptiveMCMC.RWMState(theta0, rng)

    d_obs = length(s_obs)
    s = deepcopy(s_obs); s_ = deepcopy(s_obs)

    y_dist = FT(Inf)

    # Ensure that initial distance is finite & postive:
    while !isfinite(y_dist) || y_dist < 0.0
        sim!(s, theta0, rng)
        y_dist = dist(s, s_obs)
    end

    if tol == zero(FT)
        tol = y_dist + (y_dist == zero(FT))
    end
    if adapt_tol
        step = PolynomialStepSize(stepSize_eta)
        tolerance = AdaptiveMCMC.AdaptiveScalingMetropolis(tol_targetAcceptance, one(FT), step)
        tolerance.sc[] = -log(tol)
        tols_ = zeros(FT,b)
    else
        tols_ = nothing
    end
    if typeof(cutoff) == ABCCutoff
        phi = cutoff
    elseif typeof(cutoff) == String
        if cutoff=="gaussian"
            phi = ABCGaussCutoff(tol)
        elseif cutoff=="epa"
            phi = ABCEpaCutoff(tol)
        else
            error("Unknown cutoff")
        end
    else
        phi = ABCCutoff(FT(tol), cutoff)
    end

    if adapt_cov
        if adapt_tol
            proposal = AdaptiveMCMC.AdaptiveMetropolis(theta0, 2.38/sqrt(d), step)
        else
            proposal = AdaptiveMCMC.AdaptiveMetropolis(theta0)
        end
    else
        rwm = AdaptiveMCMC.RWMState(theta0, rng)
        proposal = sc
    end

    if record_summaries
        Summary = Array{FT}(undef,d_obs,n)
    else
        Summary = missing
    end
    
    proposal_burnin = proposal
    
    y_phi = phi(y_dist)
    if y_phi == 0.0
        # If fail to accept the initial observation, enforce acceptance
        y_phi = max(phi.fun(1.0), eps(Float64)) # phi.fun(1.0)
        y_dist = phi.tol
    end
    Theta = Array{FT}(undef,d,n)
    Dist = Vector{FT}(undef,n)
 
    stats = abc_mcmc_!(Theta, Dist, Summary, tols_, s_obs, n, b, rwm, proposal, rng,
    s, s_, y_phi, y_dist, tolerance, adapt_cov, adapt_tol, proposal_burnin, record_summaries, phi, sim!, dist, prior)

    (Theta = Theta, Dist = Dist, Summary = Summary, s_obs = s_obs, cutoff = phi, 
    Stats = stats, theta0 = theta0)
end

# Worker:
function abc_mcmc_!(Theta, Dist, Summary, tols_, s_obs, n, b, rwm, proposal, rng,
    s, s_, y_phi, y_dist, tolerance, adapt_cov, adapt_tol, proposal_burnin, record_summaries, phi, sim!::Function, dist::Function, prior::Function)

    p_theta = prior(rwm.x)
    all_acc = 0; acc = 0
    for k in 1:(n+b)
        AdaptiveMCMC.draw!(rwm, proposal)
        p_theta_ = prior(rwm.y)
        sim!(s_, rwm.y, rng)
        y_dist_ = dist(s_, s_obs)
        y_phi_ = phi(y_dist_)
        α = min(one(p_theta_), exp(p_theta_ - p_theta)*y_phi_/y_phi)
        accept = (rand(rng) < α)
        if accept
            accept!(rwm)
            p_theta = p_theta_;
            y_dist = y_dist_; y_phi = y_phi_;
            s, s_ = s_, s
            all_acc += 1
            acc += (k>b)
        end
        if adapt_cov
            AdaptiveMCMC.adapt!(proposal, rwm, α, k)
        end
        if adapt_tol && k <= b
            AdaptiveMCMC.adapt!(tolerance, rwm, α, k)
            phi.tol = tols_[k] = exp(-tolerance.sc[])
        end
        if (k == b)
            proposal_burnin = deepcopy(proposal)
        end
        if (k > b)
            Theta[:,k-b] .= rwm.x
            Dist[k-b] = y_dist
            if record_summaries
                Summary[:,k-b] .= s
            end
        end
    end
    return (all_acc = all_acc/(n+b), acc = acc/n, 
       rwm = rwm, adapt = proposal, adapt_burnin = proposal_burnin, tol = tols_)
end