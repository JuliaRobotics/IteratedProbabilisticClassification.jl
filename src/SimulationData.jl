
Nor(x;m=0.0,s=1.0) = 1.0/sqrt(2*pi)/s * exp( -0.5/(s^2)*(x-m)^2 )
Nor2(x;m1=0.0,m2=0.0,s=1.0) = 0.7/sqrt(2*pi)/s * exp( -0.5/(s^2)*(x-m1)^2 ) + 0.3/sqrt(2*pi)/s * exp( -0.5/(s^2)*(x-m2)^2 )
function simdata01_true(;N1=100,N2=100)
  # common struct to hold all simulated data
  data = DataGroundTruth()
  samples = SampleData(zeros(0,0), 1, N1+N2)

  temp1 = (randn(round(Int,0.7*N1))-1.0)
  cluster1 = ([temp1; randn(N1-length(temp1))-5.0]')
  cluster2 = randn(1,N2)+1

  data.clusters[1] = cluster1
  data.clustersizes[1] = N1
  data.clusters[2] = cluster2
  data.clustersizes[2] = N2
  # one dimensional, Array{Float64,1}, with N1+N2 elements [vector alias]
  sd = [cluster1';cluster2']'
  permu = randperm(N1+N2)
  samples.samples = sd[:,permu]


y = randn(10)

  data.assignment = ([ones(Int,N1);2*ones(Int,N2)])[permu]

  # [OPTIONAL] going to carry plot layers for convenience also
  drawN = 200
  pts1 = StatsBase.sample(cluster1[1,:][:], WeightVec(1.0/N1*ones(N1)), drawN, replace=false)
  yy1 = zeros(drawN)
  for i in 1:drawN  yy1[i] = Nor2(pts1[i],m1=-1.0,m2=-5.0) end
  data.plt_lyr_cluster[1] = layer(x=pts1, y=yy1, Geom.line, Theme(default_color=colorant"blue"))[1] #parse(Colorant,"red")
  data.plt_lyr_cluster_nocolor[1] = layer(x=pts1, y=yy1, Geom.line, Theme(default_color=colorant"gray"))[1] #parse(Colorant,"red")

  pts2 = StatsBase.sample(cluster2[1,:][:], WeightVec(1.0/N2*ones(N2)), drawN, replace=false)
  yy2 = zeros(drawN)
  for i in 1:drawN  yy2[i] = Nor(pts2[i],m=+1.0) end
  data.plt_lyr_cluster[2] = layer(x=pts2, y=yy2, Geom.line, Theme(default_color=colorant"red"))[1] #parse(Colorant,"red")
  data.plt_lyr_cluster_nocolor[2] = layer(x=pts2, y=yy2, Geom.line, Theme(default_color=colorant"gray"))[1] #parse(Colorant,"red")

  return samples, data
end


# two dimensional example
function simdata02_true(;N1=100,N2=100)
  # common struct to hold all simulated data
  data = DataGroundTruth()
  samples = SampleData(zeros(0,0), 1, N1+N2)

  cluster1 = rand(MvNormal([3.0;3.0],[2.5 -1.5; -1.5 2.5]),N1)
  cluster2 = rand(MvNormal([-2.0;-2.0],[4.0 3.0; 3.0 4.0]),N2)

  data.clusters[1] = cluster1
  data.clustersizes[1] = N1
  data.clusters[2] = cluster2
  data.clustersizes[2] = N2
  # two rows, N1+N2 columns (Array{Float64,2})
  # data.samples = [cluster1';cluster2']'
  # TODO: not sure if sd should be a vector or matrix
  sd = [cluster1';cluster2']'
  permu = randperm(N1+N2)
  samples.samples = sd[:,permu]

  data.assignment = ([ones(Int,N1);2*ones(Int,N2)])[permu]

  drawN = 200

  pts1 = StatsBase.sample(cluster1[1,:][:], StatsBase.ProbabilityWeights(1.0/N1*ones(N1)), drawN, replace=false)
  data.plt_lyr_cluster[1] = layer(x=cluster1[1,:], y=cluster1[2,:], Geom.point, Theme(default_color=colorant"blue"))
  data.plt_lyr_cluster_nocolor[1] = layer(x=cluster1[1,:], y=cluster1[2,:], Geom.point, Theme(default_color=colorant"gray"))
  # data.plt_lyr_cluster[1] = layer(x=pts1, y=yy1, Geom.point, Theme(default_color=colorant"blue"))[1] #parse(Colorant,"red")
  # data.plt_lyr_cluster_nocolor[1] = layer(x=pts1, y=yy1, Geom.point, Theme(default_color=colorant"gray"))[1] #parse(Colorant,"red")


  pts2 = StatsBase.sample(cluster2[1,:][:], StatsBase.ProbabilityWeights(1.0/N2*ones(N2)), drawN, replace=false)
  data.plt_lyr_cluster[2] = layer(x=cluster2[1,:], y=cluster2[2,:], Geom.point, Theme(default_color=colorant"red"))
  data.plt_lyr_cluster_nocolor[2] = layer(x=cluster2[1,:], y=cluster2[2,:], Geom.point, Theme(default_color=colorant"gray"))
  # data.plt_lyr_cluster[2] = layer(x=pts2, y=yy2, Geom.point, Theme(default_color=colorant"red"))[1] #parse(Colorant,"red")
  # data.plt_lyr_cluster_nocolor[2] = layer(x=pts2, y=yy2, Geom.point, Theme(default_color=colorant"gray"))[1] #parse(Colorant,"red")


  return samples, data
end
