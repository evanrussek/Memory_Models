using Random
using LinearAlgebra
using Base.Threads
using Distributions
using Plots 

function expectedRtplot(η::Float64, κ::Float64, m::Int, q::Float64, T::Float64, a::Float64, Δt::Float64, reps::Int)
    tspan = 0.0:Δt:T
    len_tspan = length(tspan)
    
    x̂₀ = fill(η / m, m)
    x̂₁ = clamp.(x̂₀ + log((m - 1) * q / (1 - q)) / m * vcat(m - 1, fill(-1, m - 1)), 0, η)
    
    δ = 1 - exp(-κ * Δt)
    σ = sqrt((1 - exp(-2 * κ * Δt)) / (2 * κ))
    
    tq = Int(floor(a * T / Δt)) + 1
    
    function Xtrad_sim(X₀::Vector{Float64}, ϵ::Matrix{Float64})
        Xt = copy(X₀)
        Xtrad = zeros(len_tspan, m)
        Xtrad[1, :] = Xt

        for t in 2:tq-1
            θ_Xt = Xt .> 0.0
            sum_θ_Xt = count(θ_Xt)
            
            raw_ΔX = δ * (x̂₀ - Xt) + σ * ϵ[t, :]
            
            ΔX = θ_Xt .* raw_ΔX - (dot(θ_Xt, raw_ΔX) / sum_θ_Xt) * θ_Xt 
           
            Xt .+= ΔX
            Xtrad[t, :] = Xt
        end
        
        for t in tq:len_tspan
            θ_Xt = Xt .> 0.0
            sum_θ_Xt = count(θ_Xt)
            
            raw_ΔX = δ * (x̂₁ - Xt) + σ * ϵ[t, :]
            
            ΔX = θ_Xt .* raw_ΔX - (dot(θ_Xt, raw_ΔX) / sum_θ_Xt) * θ_Xt 
           
            Xt .+= ΔX
            Xtrad[t, :] = Xt
        end

        return Xtrad
    end

    @inline function R(x::AbstractVector{<:Real})
        exp_neg_x = exp.(-x)
        return q * (1 - exp_neg_x[1]) + ((1 - q) / (m - 1)) * sum(1 .- exp_neg_x[2:end])
    end

    function R_mean(reps::Int)
        R_sum_per_thread = zeros(len_tspan, nthreads())

        @threads for i in 1:reps
            X₀ = η * rand(Dirichlet(ones(m)))
            ϵ = randn(len_tspan, m)
        
            Xtrad_out = Xtrad_sim(X₀, ϵ)
            R_Xt = map(R, eachrow(Xtrad_out))
            
            R_sum_per_thread[:, threadid()] += R_Xt
        end

        total_R_sum = sum(R_sum_per_thread, dims=2)
        return total_R_sum[:, 1] / reps
    end

    return plot(tspan, R_mean(reps), ylim = (0,1), xlabel = "t" , ylabel = "E[R; t]")
end

# Set the input quantities
η = 4.00        # capacity
κ = 0.50        # control
m = 3           # load - number of items
q = 1.00        # cue reliability
T = 4.0         # max time
a = 0.50        # proportion of total time the retro-cue appears a*T
Δt = 0.01       # time increment
reps = 10^4     # number of replicates

expectedRtplot(η, κ, m, q, T, a, Δt, reps)
