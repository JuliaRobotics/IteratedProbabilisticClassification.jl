# debugging and plotting utilities for classification

mutable struct TuningParameters
  Mfair::Int # Cluster distribution approximation accuracy, more is better, computation becomes prohibative beyond ~1000
  EMiters::Int # expectation maximization iterations to refine Cluster distribution density estimate and sample classification, diminishing returns beyond ~30
  TuningParameters(;Mfair=200, EMiters=30) = new(Mfair, EMiters)
end

mutable struct ClassificationSystem
  numCategories::Int
  categoryLabels::Dict{Int, String}
  colors::Dict{Int, String}
  expertBelief::Dict{Int, BallTreeDensity}
  temporalBelief::Dict{Int, BallTreeDensity}
  currentBelief::Dict{Int, BallTreeDensity}
  assignment::Vector{Int}
end

mutable struct SampleData
  samples::Array{Float64,2}
  measurementDims::Int
  numSamples::Int
end

mutable struct DataGroundTruth
  assignment::Array{Int,1}
  clustersizes::Dict{Int,Int}
  clusters::Dict{Int,Array{Float64,2}}
  plt_lyr_cluster::Dict{Int, Any}
  plt_lyr_cluster_nocolor::Dict{Int, Any}
  DataGroundTruth() = new(zeros(Int, 0),  Dict{Int,Int}(),  Dict{Int,Array{Float64,2}}(),  Dict{Int,Any}(),  Dict{Int,Any}() )
  DataGroundTruth(x...) = new(x[1],x[2],x[3],x[4],x[5])
end

# Common struct to store debug information during running of the algorithm
mutable struct DebugResults
  ASSIGNED::Array{Array{Int,1},1}
  PL_MEAS::Array{Any,1}
  ACCUR_C::Dict{Int,Array{Float64,1}}
  REL_ACCUR_C::Dict{Int,Array{Float64,1}}
  INDV_MISASSIGN_C::Dict{Int,Array{Float64,1}}
  DebugResults() = new(Array{Array{Int,1},1}(),
                      [],
                      Dict{Int,Array{Float64,1}}(),
                      Dict{Int,Array{Float64,1}}(),
                      Dict{Int,Array{Float64,1}}() )
end
function defaultDebugResults()
  dbg = DebugResults()
  dbg.ACCUR_C[1] = Float64[]
  dbg.ACCUR_C[2] = Float64[]

  dbg.REL_ACCUR_C[1] = Float64[]
  dbg.REL_ACCUR_C[2] = Float64[]

  dbg.INDV_MISASSIGN_C[1] = Float64[]
  dbg.INDV_MISASSIGN_C[2] = Float64[]

  return dbg
end

mutable struct ClassificationStats
  POPFRAC::Array{Float64,2}
  SEQKLDIVERG::Dict{Int, Vector{Float64}}
  ESTBELIEF::Dict{Int, Vector{BallTreeDensity}}
  ClassificationStats() = new(zeros(0,0),
                              Dict{Int, Vector{Float64}}(),
                              Dict{Int, Vector{BallTreeDensity}}()   )
end
function defaultClassificationStats(params::TuningParameters, cs::ClassificationSystem)
  stats = ClassificationStats()
  for clbl in cs.categoryLabels
    stats.SEQKLDIVERG[clbl[1]] = Float64[]
    stats.ESTBELIEF[clbl[1]] = BallTreeDensity[]
  end
  stats.POPFRAC = zeros(cs.numCategories,params.EMiters) #Array{Float64,2}()
  stats
end



# attempt at balancing populations of different size (non-textbook, there is no textbook)
# using Dirichlet distribution as conjugate prior of categorical distribution
# a is fractions of current population classification estimates, sum(a) = 1, a.>= 0
# b is current likelihood estimate of being classified label length(b), and should ==length(a); sum(b) = 1, b.>=0
# yes, function needs a better name
function sdc2(a,b)
  aa = a.+0.6*abs(maximum(a)-minimum(a))#1#./maximum(a)
  p = rand(Dirichlet(aa))
  pp = p.*b
  pp /= sum(pp)
  rand(Categorical(pp))
end



function packDebugResults!(dbg::DebugResults, assigned, GT)

  push!(dbg.ACCUR_C[1], 100*abs(sum(assigned .== 1)-GT.clustersizes[1])/(GT.clustersizes[1]+0.0))
  push!(dbg.ACCUR_C[2], 100*abs(sum(assigned .== 2)-GT.clustersizes[2])/(GT.clustersizes[2]+0.0))

  push!(dbg.REL_ACCUR_C[1], 100*abs(sum(assigned .== 1)-GT.clustersizes[1])/(GT.clustersizes[1]+GT.clustersizes[2]))
  push!(dbg.REL_ACCUR_C[2], 100*abs(sum(assigned .== 2)-GT.clustersizes[2])/(GT.clustersizes[1]+GT.clustersizes[2]))

  push!(dbg.INDV_MISASSIGN_C[1], 100*abs(sum(assigned .== 1)-GT.clustersizes[1])/(GT.clustersizes[1]+0.0) )
  push!(dbg.INDV_MISASSIGN_C[2], 100*abs(sum(assigned .== 2)-GT.clustersizes[2])/(GT.clustersizes[2]+0.0) )

  push!(dbg.ASSIGNED, assigned)

  nothing
end



function EMClassificationRun!(
      params::TuningParameters,
      cs::ClassificationSystem,
      data::SampleData;
      debug::NothingUnion{DebugResults}=nothing,
      groundtruth::NothingUnion{DataGroundTruth}=nothing )

  # allocate some common variables
  stats = defaultClassificationStats(params, cs)
  idxs = collect(1:data.numSamples)
  weights = zeros(cs.numCategories, data.numSamples)

  ### [1.0] Classification algorithm part 1 of 2 -- the measurement, generalized expectation maximization (EM) procedure
  # [1.1] loop over cluster assignment sample drawing process
  for m in 1:params.EMiters # EM iterations per measurement cycle, this is a tuning parameter -> more iterations should assymptote to true assignment distribution
    # [1.2] numerically approximate a Bayesian prior -- (interpret as Dirichlet sampling process with some implicit concentration parameter)

    # find the joint distribution using expert prior, EM measurement iteration (HMM prediction not shown in this example yet)
    if m>1  for clbl in cs.categoryLabels
      cs.currentBelief[clbl[1]] = cs.currentBelief[clbl[1]] * cs.expertBelief[clbl[1]]
    end end

    # evaluate the point likelihood of being in each cluster
    for i in 1:cs.numCategories
      weights[i,:] = evaluateDualTree(cs.currentBelief[i], data.samples)
    end

    # [1.3] assign cluster as sampling from categorical distribution
    popFrac = [sum(cs.assignment .== 1); sum(cs.assignment .== 2)]/(data.numSamples+0.0)
    stats.POPFRAC[:,m] = popFrac'
    for i in 1:data.numSamples
      p = vec(weights[:,i])
      p = p/sum(p) # renormalize to valid probability distribution
      sel = false ?   rand( Categorical( p ) )   :   round(Int,sdc2(popFrac, p))
      cs.assignment[i] =  sel
    end

    # build new intermediate cluster density estimates
    # but subsample to fair chance and practically computable sets first
    clusters = Dict{Int, Array{Float64,2}}()
    subSampledClusters = Dict{Int, Array{Float64,2}}()
    for clbl in cs.categoryLabels
      msk = cs.assignment .== clbl[1]
      @show size(msk), size(data.samples)
      clusters[clbl[1]] = data.samples[:, msk ]
      len = size(clusters[clbl[1]],2)

      useidx = StatsBase.sample(idxs[1:size(clusters[clbl[1]],2)], ProbabilityWeights(1.0/len*ones(len)), params.Mfair, replace=false)
      subSampledClusters[clbl[1]] = clusters[clbl[1]][:,useidx]

      newBel = kde!(subSampledClusters[clbl[1]])
      diverg = kld(newBel, cs.currentBelief[clbl[1]])[1]
      cs.currentBelief[clbl[1]] = newBel
      push!(stats.SEQKLDIVERG[clbl[1]], diverg)
      push!(stats.ESTBELIEF[clbl[1]], deepcopy(cs.currentBelief[clbl[1]]))
    end

    # measure classification accuracy
    if debug != nothing && groundtruth != nothing
      packDebugResults!(debug, cs.assignment, groundtruth)
    end
  end
  stats
end





function dispersePoints(X::Array{Float64,2}, dispersion)
  r, c = size(X)
  Y = zeros(r,c)
  for i in 1:c
    Y[:,i] = X[:,i] + rand(dispersion)
  end
  return Y
end












#
