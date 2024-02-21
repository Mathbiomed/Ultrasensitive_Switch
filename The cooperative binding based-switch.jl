using Random # randexp()
using StatsBase # Weights() and sample()
using Plots
using LinearAlgebra
using Statistics
using Distributions
using Roots
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
function tranHill3(krrt,coop)
    return 1 - 1/(1 + 3*coop^2*krrt + 3*coop^3*krrt^2 + coop^3*krrt^3) # Transcriptional activity of the cooperative binding-based switch
end

norma1 = 10.0^0.0
normb1 = 10.0^-2.0
coop1 = 10.0^-4.0
coop2 = 10.0^-4.0
tend = 20000.0
tranV=0.9
f(x)=tranHill3(x,coop1)-tranV
krrt=find_zeros(f,0,10^6)[1] # The effective nnumber of repressors where the transcriptional activity has the value of tranV

parameterfull = (a1=norma1, b1=normb1, stoich=[[1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, 0], [0, -1, 1, 0, 0, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, -1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, -1, 0, 0, 0], [0, 0, -1, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, -1, 0, 0], [0, 0, 0, -1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, -1, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, -1, 0], [0, 0, 0, 0, -1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, -1, 0, 0], [0, 0, 0, 0, -1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, -1, 0], [0, 0, 0, 0, 0, -1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, -1], [0, 0, 0, 0, 0, 0, -1, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0, -1], [0, 0, 0, 0, 0, 0, 0, -1, 1], [0, 0, 0, 0, 0, 0, 0, 1, -1],])
# parameter values and stoichometry matrix

num=1000
hill3(u, p, t) = [p.a1*u[2], p.a1*u[3], p.a1*u[4], p.a1*u[5], p.a1*u[6], p.a1*u[7], p.a1*u[8], p.b1*u[1], u[2], krrt*u[3], u[2], krrt*u[4], u[2], krrt*u[5], u[3], coop1*krrt*u[6], u[3], coop1*krrt*u[7], u[4], coop1*krrt*u[6], u[4], coop1*krrt*u[8], u[5], coop1*krrt*u[7], u[5], coop1*krrt*u[8], u[6], coop2*coop1*krrt*u[9], u[7], coop2*coop1*krrt*u[9], u[8], coop2*coop1*krrt*u[9]]
s0 = [0, 1, 0, 0, 0, 0, 0, 0, 0] # M, E000, E001, E010, E100, E011, E101, E110, E111
global saveM = zeros(Float64,num,numpt+1)
global saveE000 = zeros(Float64,num,numpt+1)
global saveE001 = zeros(Float64,num,numpt+1)
global saveE010 = zeros(Float64,num,numpt+1)
global saveE100 = zeros(Float64,num,numpt+1)
global saveE011 = zeros(Float64,num,numpt+1)
global saveE101 = zeros(Float64,num,numpt+1)
global saveE110 = zeros(Float64,num,numpt+1)
global saveE111 = zeros(Float64,num,numpt+1)
for iter in 1:num
    hill3sol = ssa_direct(hill3, s0, tend, parameterfull, parameterfull.stoich)
    saveM[iter,:] = (hill3sol.u[1,:])
    saveE000[iter,:] = (hill3sol.u[2,:])
    saveE001[iter,:] = (hill3sol.u[3,:])
    saveE010[iter,:] = (hill3sol.u[4,:])
    saveE100[iter,:] = (hill3sol.u[5,:])
    saveE011[iter,:] = (hill3sol.u[6,:])
    saveE101[iter,:] = (hill3sol.u[7,:])
    saveE110[iter,:] = (hill3sol.u[8,:])
    saveE111[iter,:] = (hill3sol.u[9,:])
    println(iter)
end

writedlm("mRNAs.csv", Tables.table(transpose(saveM)), ',')
# Export the simulated number of mRNAs
