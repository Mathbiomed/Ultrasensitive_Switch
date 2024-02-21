using Random # randexp()
using StatsBase # Weights() and sample()
using Plots
using LinearAlgebra
using Statistics
using Distributions
using DelimitedFiles, Tables

#Stochastic chemical reaction: Gillespie Algorithm
Random.seed!(1)
global numpt = 10000
function ssa_direct(model, u0, tend, p, stoich; tstart=zero(tend))
    t = tstart   # Current time
    ts = zeros(Float64,1,numpt+1)     # Time points
    u = copy(u0) # Current state
    us = zeros(Float64,size(stoich[1],1),numpt+1) # Record of states
    us[:,1] = u
    tind = 1
    while t < tend
        a = model(u, p, t)               # propensities
        dt = -log(rand())/ sum(a)          # Time step
        du = sample(stoich, Weights(a))  # Choose the stoichiometry for the next reaction
        t += dt   # Update time
        if (t <= tend)
            if (t > tind*tend/numpt)
                us[:,tind+1:Int(floor(t*numpt/tend))+1]=reduce(hcat,[u for i in 1:(Int(floor(t*numpt/tend)-tind)+1)])
            end
        else
            us[:,tind+1:numpt+1]=reduce(hcat,[u for i in 1:(numpt-tind+1)])
        end
        u .+= du  # Update state
        tind = Int(floor(t*numpt/tend))+1 # Update index
    end
    return (t = ts, u = us)
end

using DifferentialEquations
function normfreeR(rat,ksat)
    return (rat .- 1 .- ksat .+ sqrt.((1 .- rat .- ksat) .^2 .+4*ksat))/2
end
function normfreeA(rat,ksat)
    return (1 .- rat .- ksat .+ sqrt.((1 .- rat .- ksat) .^2 .+4*ksat))/2
end
function tranBS(rat,kaat,ksat,kbat)
    return (normfreeA(rat,ksat) ./ kaat)/(1 .+ (normfreeA(rat,ksat) ./ kaat) .+ (normfreeA(rat,ksat) ./ kaat)*(normfreeR(rat,ksat) ./ kbat))
end

tend = 20000.0
norma1=10.0^0.0
normb1=10.0^-2.0
kaat=10.0^-3.0
ksat=10.0^-5.0
kbat=10.0^-3.0
rat=10^-0.1

parameterfull = (a1=norma1, b1=normb1, stoich=[[1, 0, 0, 0], [-1, 0, 0, 0], [0, -1, 1, 0], [0, 1, -1, 0], [0, 0, -1, 1], [0, 0, 1, -1]])
# parameter values and stoichometry matrix

num=1000
blockseqfull(u, p, t) = [p.a1*u[3], p.b1*u[1], normfreeA(rat,ksat)*u[2], kaat*u[3],normfreeR(rat,ksat)*u[3], kbat*u[4]]
s0 = [0, 1, 0, 0] # M, EF, EA, ER

global saveM = zeros(Float64,num,numpt+1)
global saveEF = zeros(Float64,num,numpt+1)
global saveEA = zeros(Float64,num,numpt+1)
global saveER = zeros(Float64,num,numpt+1)

for iter in 1:num
    blockseq = ssa_direct(blockseqfull, s0, tend, parameterfull, parameterfull.stoich)
    saveM[iter,:] = (blockseq.u[1,:])
    saveEF[iter,:] = (blockseq.u[2,:])
    saveEA[iter,:] = (blockseq.u[3,:])
    saveER[iter,:] = (blockseq.u[4,:])
    if iter%100==0
        println(iter)
    end
end

writedlm("mRNAs.csv", Tables.table(transpose(saveM)), ',')
# Export the simulated number of mRNAs
