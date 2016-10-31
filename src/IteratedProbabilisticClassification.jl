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
	classifyOneStep,
	classifyConverge,
	plotClassification1D,
	plotClassificationStats,
	plotPopulationFraction,
	packDebugResults!,

	# functions under development
	sdc2,
	# batchClassifyConverge

	# previous code
	simdata01_true,
	simdata02_true,
	DebugResults,
	defaultDebugResults,
	DataGroundTruth,
	SampleData,
	PALETTE


include("ClassificationUtilities.jl")

PALETTE = ["deepskyblue";"red";"magenta";"black";"green";"blue"]

function plotMarginalClassificationBelief()
	pl_init = plotKDE([c1_expert,c2_expert],c=PALETTE)
	lyr_smpls = layer(x=samples, y=zeros(size(samples)), Geom.point)
	push!(pl_init.layers, lyr_smpls[1] )
	push!(pl_init.layers, lyr_smpls_trueA[1])
	push!(pl_init.layers, lyr_smpls_trueB[1])

	#Gadfly.draw(PDF("InitialClusterDist.pdf",20cm,12cm),pl_init)

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


		nothing
end



end # module
