module IteratedProbabilisticClassification

using
	KernelDensityEstimate,
	Distributions

export
	classifyOneStep,
	classifyConverge,

	# function under development 
	# batchClassifyConverge

include("ClassificationUtilities.jl")


end
