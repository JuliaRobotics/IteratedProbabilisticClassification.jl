# debugging and plotting utilities for classification

type TuningParameters
  Mfair::Int # Cluster distribution approximation accuracy, more is better, computation becomes prohibative beyond ~1000
  EMiters::Int # expectation maximization iterations to refine Cluster distribution density estimate and sample classification, diminishing returns beyond ~30
  TuningParameters(;Mfair=200, EMiters=30) = new(Mfair, EMiters)
end

type ClassificationSystem
  expertBelief::Dict{UTF8String, BallTreeDensity}
  temporalBelief::Dict{UTF8String, BallTreeDensity}
  currentBelief::Dict{UTF8String, BallTreeDensity}
end

type SimData
  samples::Array{Float64,2}
  clustersizes::Dict{Int,Int}
  clusters::Dict{Int,Array{Float64,2}}
  plt_lyr_cluster::Dict{Int, Any}
  plt_lyr_cluster_nocolor::Dict{Int, Any}
  SimData() = new(zeros(0,2),  Dict{Int,Int}(),  Dict{Int,Array{Float64,2}}(),  Dict{Int,Any}(),  Dict{Int,Any}() )
  Simdata(x...) = new(x[1],x[2],x[3],x[4],x[5])
end

# Common struct to store debug information during running of the algorithm
type DebugResults
  ASSIGNED::Array{Array{Int,1},1}
  PL_MEAS::Array{Any,1}
  ACCUR_C::Dict{Int,Array{Float64,1}}
  REL_ACCUR_C::Dict{Int,Array{Float64,1}}
  INDV_MISASSIGN_C::Dict{Int,Array{Float64,1}}
  POPFRAC::Array{Float64,2}
  ESTBELIEFS::Vector{Dict{UTF8String, BallTreeDensity}}
  DebugResults() = new(Array{Array{Int,1},1}(), [],  Dict{Int,Array{Float64,1}}(),  Dict{Int,Array{Float64,1}}(),  Dict{Int,Array{Float64,1}}(), zeros(0,0), Vector{Dict{UTF8String, BallTreeDensity}}() )
end

Nor(x;m=0.0,s=1.0) = 1.0/sqrt(2*pi)/s * exp( -0.5/(s^2)*(x-m)^2 )
Nor2(x;m1=0.0,m2=0.0,s=1.0) = 0.7/sqrt(2*pi)/s * exp( -0.5/(s^2)*(x-m1)^2 ) + 0.3/sqrt(2*pi)/s * exp( -0.5/(s^2)*(x-m2)^2 )
function simdata01_true(;N1=100,N2=100)
  # common struct to hold all simulated data
  data = SimData()

  temp1 = (randn(round(Int,0.7*N1))-1.0)
  cluster1 = ([temp1; randn(N1-length(temp1))-5.0]')
  cluster2 = randn(1,N2)+1

  data.clusters[1] = cluster1
  data.clustersizes[1] = N1
  data.clusters[2] = cluster2
  data.clustersizes[2] = N2
  # one dimensional, Array{Float64,1}, with N1+N2 elements [vector alias]
  data.samples = [cluster1';cluster2']'

  # [OPTIONAL] going to carry plot layers for convenience also
  drawN = 200
  pts1 = StatsBase.sample(cluster1[1,:][:], WeightVec(1.0/N1*ones(N1)), drawN, replace=false)
  yy1 = zeros(drawN)
  for i in 1:drawN  yy1[i] = Nor2(pts1[i],m1=-1.0,m2=-5.0) end
  data.plt_lyr_cluster[1] = layer(x=pts1, y=yy1, Geom.line, Theme(default_color=colorant"blue"))[1] #parse(Colorant,"red")
  data.plt_lyr_cluster_nocolor[1] = layer(x=pts1, y=yy1, Geom.line, Theme(default_color=colorant"gray"))[1] #parse(Colorant,"red")
  # yy1 = zeros(N1)
  # for i in 1:N1  yy1[i] = Nor2(cluster1[1,i],m1=-1.0,m2=-5.0) end
  # data.plt_lyr_cluster[1] = layer(x=cluster1[1,:][:], y=yy1, Geom.point, Theme(default_color=colorant"blue"))[1] #parse(Colorant,"red")
  # data.plt_lyr_cluster_nocolor[1] = layer(x=cluster1[1,:][:], y=yy1, Geom.point, Theme(default_color=colorant"gray"))[1] #parse(Colorant,"red")

  pts2 = StatsBase.sample(cluster2[1,:][:], WeightVec(1.0/N2*ones(N2)), drawN, replace=false)
  yy2 = zeros(drawN)
  for i in 1:drawN  yy2[i] = Nor(pts2[i],m=+1.0) end
  data.plt_lyr_cluster[2] = layer(x=pts2, y=yy2, Geom.line, Theme(default_color=colorant"red"))[1] #parse(Colorant,"red")
  data.plt_lyr_cluster_nocolor[2] = layer(x=pts2, y=yy2, Geom.line, Theme(default_color=colorant"gray"))[1] #parse(Colorant,"red")
  # yy2 = zeros(N2)
  # for i in 1:N2  yy2[i] = Nor(cluster2[1,i],m=+1.0) end
  # data.plt_lyr_cluster[2] = layer(x=cluster2[1,:][:], y=yy2, Geom.point, Theme(default_color=colorant"red"))[1] #parse(Colorant,"red")
  # data.plt_lyr_cluster_nocolor[2] = layer(x=cluster2[1,:][:], y=yy2, Geom.point, Theme(default_color=colorant"gray"))[1] #parse(Colorant,"red")

  return data
end


# two dimensional example
function simdata02_true(;N1=100,N2=100)
  # common struct to hold all simulated data
  data = SimData()

  cluster1 = rand(MvNormal([3.0;3.0],[[2.5;-1.5]';[-1.5;2.5]']),N1)
  cluster2 = rand(MvNormal([-2.0;-2.0],[[4.0;3.0]';[3.0;4.0]']),N2)

  data.clusters[1] = cluster1
  data.clustersizes[1] = N1
  data.clusters[2] = cluster2
  data.clustersizes[2] = N2
  # two rows, N1+N2 columns (Array{Float64,2})
  data.samples = [cluster1';cluster2']'

  data.plt_lyr_cluster[1] = layer(x=cluster1[1,:], y=cluster1[2,:], Geom.point, Theme(default_color=colorant"blue"))
  data.plt_lyr_cluster_nocolor[1] = layer(x=cluster1[1,:], y=cluster1[2,:], Geom.point, Theme(default_color=colorant"gray"))
  data.plt_lyr_cluster[2] = layer(x=cluster2[1,:], y=cluster2[2,:], Geom.point, Theme(default_color=colorant"red"))
  data.plt_lyr_cluster_nocolor[2] = layer(x=cluster2[1,:], y=cluster2[2,:], Geom.point, Theme(default_color=colorant"gray"))

  return data
end


function plotClassificationStats(params, dbg)
  pl_accur = plot(
  layer(x=1:params.EMiters, y=dbg.ACCUR_C[1], Geom.line, Theme(default_color=colorant"blue" )),
  layer(x=1:params.EMiters, y=dbg.ACCUR_C[2], Geom.line, Theme(default_color=colorant"red"  )),
  Guide.title("Absolute percentage error in sample count among classifications"),
  Guide.ylabel("%")
  );

  pl_rel_accur = plot(
  layer(x=1:params.EMiters, y=dbg.REL_ACCUR_C[1], Geom.line, Theme(default_color=colorant"blue" )),
  layer(x=1:params.EMiters, y=dbg.REL_ACCUR_C[2], Geom.line, Theme(default_color=colorant"red"  )),
  Guide.title("Relative percentage error in sample count among classifications"),
  Guide.ylabel("%")
  );

  pl_indvmissassign = plot(
  layer(x=1:params.EMiters, y=dbg.INDV_MISASSIGN_C[1], Geom.line, Theme(default_color=colorant"blue" )),
  layer(x=1:params.EMiters, y=dbg.INDV_MISASSIGN_C[2], Geom.line, Theme(default_color=colorant"red"  )),
  # layer(x=1:params.EMiters, y=abs(dbg.INDV_MISASSIGN_C[1] - dbg.INDV_MISASSIGN_C[2]), Geom.line, Theme(default_color=colorant"magenta"  )),
  Guide.title("Common percentage error count among classifications (unweighted)"),
  Guide.ylabel("%")
  );

  pl_popfrac = plot(
  layer(x=1:params.EMiters, y=dbg.POPFRAC[1,:], Geom.line, Theme(default_color=colorant"blue" )),
  layer(x=1:params.EMiters, y=dbg.POPFRAC[2,:], Geom.line, Theme(default_color=colorant"red"  )),
  # layer(x=1:params.EMiters, y=abs(dbg.INDV_MISASSIGN_C[1] - dbg.INDV_MISASSIGN_C[2]), Geom.line, Theme(default_color=colorant"magenta"  )),
  Guide.title("Population fraction estimates"),
  Guide.ylabel("%")
  );

  vstack(pl_accur, pl_rel_accur, pl_indvmissassign, pl_popfrac)
end

# attempt at balancing populations of different size (non-textbook, there is no textbook)
# using Dirichlet distribution as conjugate prior of categorical distribution
# a is fractions of current population classification estimates, sum(a) = 1, a.>= 0
# b is current likelihood estimate of being classified label length(b), and should ==length(a); sum(b) = 1, b.>=0
# yes, function needs a better name
function sdc2(a,b)
  aa = a+0.7*abs(maximum(a)-minimum(a))#1#./maximum(a)
  p = rand(Dirichlet(aa))
  pp = p.*b
  pp /= sum(pp)
  rand(Categorical(pp))
end
