# debugging and plotting utilities for classification

type TuningParameters
  Mfair::Int # Cluster distribution approximation accuracy, more is better, computation becomes prohibative beyond ~1000
  EMiters::Int # expectation maximization iterations to refine Cluster distribution density estimate and sample classification, diminishing returns beyond ~30
  TuningParameters(;Mfair=200, EMiters=30) = new(Mfair, EMiters)
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
  DebugResults() = new(Array{Array{Int,1},1}(), [],  Dict{Int,Array{Float64,1}}(),  Dict{Int,Array{Float64,1}}(),  Dict{Int,Array{Float64,1}}() )
end

Nor(x;m=0.0,s=1.0) = 1.0/sqrt(2*pi)/s * exp( -0.5/(s^2)*(x-m)^2 )
Nor2(x;m1=0.0,m2=0.0,s=1.0) = 0.7/sqrt(2*pi)/s * exp( -0.5/(s^2)*(x-m1)^2 ) + 0.3/sqrt(2*pi)/s * exp( -0.5/(s^2)*(x-m2)^2 )
function simdata01(;N1=100,N2=100)
  # common struct to hold all simulated data
  data = SimData()

  temp1 = (randn(round(Int,0.7*N1))-1.0)
  cluster1 = [temp1; randn(N1-length(temp1))-5.0]
  cluster2 = randn(N2)+1

  data.clusters[1] = cluster1
  data.clustersizes[1] = N1
  data.clusters[2] = cluster2
  data.clustersizes[2] = N2
  # one dimensional, Array{Float64,1}, with N1+N2 elements [vector alias]
  data.samples = [cluster1;cluster2]

  # [OPTIONAL] going to carry plot layers for convenience also
  yy1 = zeros(N1)
  for i in 1:N1  yy1[i] = Nor2(cluster1[i],m1=-1.0,m2=-5.0) end
  data.plt_lyr_cluster[1] = layer(x=cluster1, y=yy1, Geom.point, Theme(default_color=colorant"blue")) #parse(Colorant,"red")
  data.plt_lyr_cluster_nocolor[1] = layer(x=cluster1, y=yy1, Geom.point, Theme(default_color=colorant"gray")) #parse(Colorant,"red")

  yy2 = zeros(N2)
  for i in 1:N2  yy2[i] = Nor(cluster2[i],m=+1.0) end
  data.plt_lyr_cluster[2] = layer(x=cluster2, y=yy2, Geom.point, Theme(default_color=colorant"red")) #parse(Colorant,"red")
  data.plt_lyr_cluster_nocolor[2] = layer(x=cluster2, y=yy2, Geom.point, Theme(default_color=colorant"gray")) #parse(Colorant,"red")

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
