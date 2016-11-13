module IteratedProbabilisticClassification

using
	KernelDensityEstimate,
	Distributions,
	StatsBase,
	Gadfly

export
	# Major types and functions used for classification
	ClassificationSystem,
	TuningParameters,
	ClassificationStats,
	defaultClassificationStats,
	classifyOneStep,
	classifyConverge,
	plotClassification1D,
	plotClassificationStats,
	plotEMStatus,
	packDebugResults!,
	EMClassificationRun!,
	dispersePoints,

	# functions under development
	sdc2,
	plotUtil1D,
	plotUtil2D,
	# batchClassifyConverge

	# previous code
	simdata01_true,
	simdata02_true,
	DebugResults,
	defaultDebugResults,
	DataGroundTruth,
	SampleData,
	PALETTE


PALETTE = ["deepskyblue";"red";"magenta";"black";"green";"blue"]


typealias VoidUnion{T} Union{Void, T}


include("ClassificationUtilities.jl")
include("PlottingUtilities.jl")
include("SimulationData.jl")




function plotMarginalClassificationBelief()
	pl_init = plotKDE([c1_expert,c2_expert],c=PALETTE)
	lyr_smpls = layer(x=samples, y=zeros(size(samples)), Geom.point)
	push!(pl_init.layers, lyr_smpls[1] )
	push!(pl_init.layers, lyr_smpls_trueA[1])
	push!(pl_init.layers, lyr_smpls_trueB[1])

	#Gadfly.draw(PDF("InitialClusterDist.pdf",20cm,12cm),pl_init)
	error("Incomplete function")
end

function plotClassification1D(
			;groundtruth=nothing,
			samples::Array{Float64,2}=Array{Float64,2}(),
			belief::BallTreeDensity=nothing)
		dims = size(samples, 1)

		belief == nothing ? nothing : (dims == Ndim(belief) ? nothing : error("belief and sample dimensions don't match") )

		plr = ceil(Int, sqrt(dims))
		plc = ceil(Int, dims/plr)

		PL = Gadfly.Context[]
		for i in dims
			# pl =
			push!(PL, pl)
		end

		error("Incomplete function")

		nothing
end











end # module
