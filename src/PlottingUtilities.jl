


function layersGroundTruth1D(gt::DataGroundTruth)
	PL = []
	push!(PL, gt.plt_lyr_cluster[1])
	push!(PL, gt.plt_lyr_cluster[2])
	PL
end

function layersGroundTruth2D(id::Int, gt::DataGroundTruth)
	PL = Gadfly.Context[]
	PL = union(PL, gt.plt_lyr_cluster[id])
	PL
end
function layersGroundTruth2D(ids::Vector{Int}, gt::DataGroundTruth)
	PL = Gadfly.Context[]
	for id in ids
		PL = union(PL, gt.plt_lyr_cluster[id])
	end
	PL
end


function layersExpertBelief1D(cs::ClassificationSystem;
															margdim=1,
															colors::NothingUnion{Dict{Int, String}}=nothing )
	# use defaults
	colors = colors == nothing ? cs.colors : colors

  # prep data
	pr, cl = BallTreeDensity[], String[]
	for p in cs.expertBelief
		push!(pr, marginal(p[2],[margdim]))
		push!(cl, colors[p[1]]) # get accompanying color
	end

	# plot and return layers
	pl_init = plotKDE(pr,c=cl);
	pl_init.layers
end

function layersExpertBelief2D(id::Int, cs::ClassificationSystem;
															margdim=1,
															colors::NothingUnion{Dict{Int, String}}=nothing )
	# use defaults
	colors = colors == nothing ? cs.colors : colors

  # prep data
	pr, cl = BallTreeDensity[], String[]
	for p in cs.expertBelief
		push!(pr, marginal(p[2],[margdim]))
		push!(cl, colors[p[1]]) # get accompanying color
	end

	# plot and return layers
	pl_init = plotKDE(pr,c=cl);
	pl_init.layers
end

function layersCurrentBelief1D(cs::ClassificationSystem;
															margdim=1,
															colors::NothingUnion{Dict{Int, String}}=nothing )
	# use defaults
	colors = colors == nothing ? cs.colors : colors

  # prep data
	pr, cl = BallTreeDensity[], String[]
	for p in cs.currentBelief
		push!(pr, marginal(p[2],[margdim]))
		push!(cl, colors[p[1]]) # get accompanying color
	end

	# plot and return layers
	pl_init = plotKDE(pr,c=cl);
	pl_init.layers
end

function plotUtil1D(;
		sampledata::NothingUnion{SampleData}=nothing,
		groundtruth::NothingUnion{DataGroundTruth}=nothing,
		cs::NothingUnion{ClassificationSystem}=nothing,
		expertcolor::NothingUnion{Dict{Int,String}}=nothing,
		drawcurrent::Bool=false,
		currentcolor::NothingUnion{Dict{Int,String}}=nothing
		)

		PL = []
		PL = groundtruth == nothing ? PL : union(PL, layersGroundTruth1D(groundtruth))
		# push!(PL, Guide.title("True classification for intersecting clusters in 1 dimension"))

		PL = cs == nothing ? PL : union(PL, layersExpertBelief1D(cs, colors=expertcolor))

		PL = (cs == nothing || !drawcurrent) ? PL : union(PL, layersCurrentBelief1D(cs, colors=currentcolor))

		# layer(x=data.samples,y=zeros(size(data.samples)),Geom.point,Theme(default_color=colorant"gray")),

		plot(PL...)
end


function plotUtil2D(id::Union{Int, Vector{Int}};
		sampledata::NothingUnion{SampleData}=nothing,
		groundtruth::NothingUnion{DataGroundTruth}=nothing,
		cs::NothingUnion{ClassificationSystem}=nothing,
		expertcolor::NothingUnion{Dict{Int,String}}=nothing,
		drawcurrent::Bool=false,
		currentcolor::NothingUnion{Dict{Int,String}}=nothing
		)

		PL = []
		PL = groundtruth == nothing ? PL : union(PL, layersGroundTruth2D(id, groundtruth))
		# push!(PL, Guide.title("True classification for intersecting clusters in 1 dimension"))

		PL = cs == nothing ? PL : union(PL, layersExpertBelief2D(id, cs, colors=expertcolor))
		#
		# PL = (cs == nothing || !drawcurrent) ? PL : union(PL, layersCurrentBelief1D(cs, colors=currentcolor))

		plot(PL...)
end







function plotEMStatus(params, stats)
  plfrac = plot(
  layer(x=1:params.EMiters, y=stats.POPFRAC[1,:], Geom.line, Theme(default_color=colorant"blue" )),
  layer(x=1:params.EMiters, y=stats.POPFRAC[2,:], Geom.line, Theme(default_color=colorant"red"  )),
  # layer(x=1:params.EMiters, y=abs(dbg.INDV_MISASSIGN_C[1] - dbg.INDV_MISASSIGN_C[2]), Geom.line, Theme(default_color=colorant"magenta"  )),
  Guide.title("Population fraction estimates"),
  Guide.ylabel("%")
  )

	plkl = plot(
	layer(x=1:params.EMiters, y=abs.(stats.SEQKLDIVERG[1][:]), Geom.line, Theme(default_color=colorant"blue" )),
	layer(x=1:params.EMiters, y=abs.(stats.SEQKLDIVERG[2][:]), Geom.line, Theme(default_color=colorant"red"  )),
	# layer(x=1:params.EMiters, y=abs(dbg.INDV_MISASSIGN_C[1] - dbg.INDV_MISASSIGN_C[2]), Geom.line, Theme(default_color=colorant"magenta"  )),
	Guide.title("Abs sequential KL divergence, | D(p_{k+1} || p_{k}) |"),
	)

	vstack(plfrac, plkl)
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

  # pl_indvmissassign = plot(
  # layer(x=1:params.EMiters, y=dbg.INDV_MISASSIGN_C[1], Geom.line, Theme(default_color=colorant"blue" )),
  # layer(x=1:params.EMiters, y=dbg.INDV_MISASSIGN_C[2], Geom.line, Theme(default_color=colorant"red"  )),
  # # layer(x=1:params.EMiters, y=abs(dbg.INDV_MISASSIGN_C[1] - dbg.INDV_MISASSIGN_C[2]), Geom.line, Theme(default_color=colorant"magenta"  )),
  # Guide.title("Common percentage error count among classifications (unweighted)"),
  # Guide.ylabel("%")
  # );

  vstack(pl_accur, pl_rel_accur)
end
