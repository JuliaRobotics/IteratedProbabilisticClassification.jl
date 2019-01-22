# classification demo

# avoid unrelated libz error
# using ImageMagick
# using Makie

using KernelDensityEstimate
using Gadfly
using Distributions
using StatsBase, Random
using IteratedProbabilisticClassification
# don't mind syntax warnings, these are being removed from package dependencies over time


# modified EM classification algorithm has these major tuning parameter
global localparams = TuningParameters(
  Mfair=300,   # Cluster distribution approximation accuracy, more is better: 50 is bare minium, and computation slow above ~1000
  EMiters=40   # expectation maximization iterations to refine Cluster belief estimate classification assignments: 5 is minimum, and diminishing returns beyond ~70
)

# Example: simulated data with ground truth for reference
N1, N2 = 5000, 1000
# N1, N2 = 4000, 3000
data, groundtruth = simdata02_true(N1=N1,N2=N2)

# actual distribution of data
Gadfly.set_default_plot_size(15cm, 10cm)
# what does the data look like
Gadfly.plot(x=data.samples[1,:],y=data.samples[2,:], Geom.histogram2d)



# Ground truth distribution of underlying data clusters
# Gadfly.set_default_plot_size(15cm, 10cm)
# a generic 1D plotting tool for IteratedProbablisticClassification
plotUtil2D([2;1], groundtruth=groundtruth)




# belief from expert prediction, also used for initialization
c1_expt_pts = rand(MvNormal([5.0;5.0],[3.0 -1.0; -1.0 3.0]),100)
c1_expert = kde!(reshape([5.0,5.0],2,1), [3.0,3.0]) # Pretty much a normal distribution
c1_expert = kde!(c1_expt_pts)
c2_expert = kde!(reshape([-5.0,-6.0],2,1), [4.0,4.0]) # Pretty much a normal distribution

# cs is the main structure which classification algorithm will operate on, and modify during execution
cs = ClassificationSystem(
  2,                              # number of categories or classes available for assignment
  Dict(1=>"class1", 2=>"class2"), # names of classes
  Dict(1=>"blue", 2=>"red"),      # default plotting colors
  Dict(1=>c1_expert, 2=>c2_expert), # user forcing behaviour (expert guidance); required
  Dict(1=>deepcopy(c1_expert), 2=>deepcopy(c2_expert)), # initialize temporal prediction (forward-backward smoothing)
  Dict(1=>deepcopy(c1_expert), 2=>deepcopy(c2_expert)), # initialize current belief (0 iterations)
  rand(Categorical([0.5;0.5]),length(data.samples)) # initialize samples with random assignment, 50/50% in this 2 class case
);
println()


# expert may modify expertBelief using output from previous run of classification algorithm
# pts = getPoints(cs.currentBelief[1])
# cs.expertBelief[1] = kde!(dispersePoints(pts, MvNormal([-1.0],[1.0]) ))
# pts = getPoints(cs.currentBelief[2])
# cs.expertBelief[2] = kde!(dispersePoints(pts, MvNormal([2.0],[1.0]) ))
# println()

Gadfly.set_default_plot_size(20cm, 7cm)
# plotUtil1D(cs=cs) # if we didn't have gt
plotUtil2D(1,cs=cs)



# simulation data allows us to debug with absolute knowledge
dbg = defaultDebugResults()

# do the classification
stats = EMClassificationRun!(localparams, cs, data, debug=dbg, groundtruth=groundtruth);
println()


Gadfly.set_default_plot_size(20cm, 10cm)
plotUtil1D(cs=cs, groundtruth=groundtruth,
  drawcurrent=true,
  expertcolor=Dict(1=>"gray",2=>"gray")
)

sum(cs.assignment .== 1), sum(cs.assignment .== 2)



Gadfly.set_default_plot_size(20cm, 10cm)
plot(
layer(x=data.samples[1,cs.assignment .== 2], Geom.histogram,Theme(default_color=colorant"red")),
layer(x=data.samples[1,cs.assignment .== 1], Geom.histogram)
)



# proxy to convergence of classification algorithm, always available from stats structure
Gadfly.set_default_plot_size(20cm, 15cm)
plotEMStatus(localparams,stats)


# only available when ground truth data is available
Gadfly.set_default_plot_size(20cm, 15cm)
plotClassificationStats(localparams, dbg)

# pts = getPoints(cs.expertBelief[1])
# pts2 = dispersePoints(pts, MvNormal([0.0],[3.0]) )
# p2 = kde!(pts2)
# plotKDE([cs.expertBelief[1];p2])


#
