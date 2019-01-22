

## Loading FCB test data

# must be imported first to avoid unrelated libz error
using ImageMagick

using HDF5, MAT

datadir = joinpath(ENV["HOME"], "data", "fcb")

file = matopen(joinpath(datadir, "test_fcb_col_notes.mat"))
fcbCols = read(file, "fcb_data_col_notes")
close(file)

file = matopen(joinpath(datadir, "test_fcb_class.mat"))
fcbClass = read(file, "fcb_class")
close(file)

file = matopen(joinpath(datadir, "test_fcb_time.mat"))
fcbTime = read(file, "time")
close(file)

file = matopen(joinpath(datadir, "test_fcb_data.mat"))
fcbData = read(file, "fcb_data")
close(file)



## Basic plotting of data
using Makie

# don't open a separate Makie plot window -- use atom plot pane instead
AbstractPlotting.inline!(true)

## Makie basic plotting example
# generate some data
# x = rand(10)
# y = rand(10)
# colors = rand(10)
# scene = scatter(x, y, color = colors, markersize = 0.05*ones(length(x)))
# scene = scatter(x, y, markersize = 0.05*ones(length(x)))



frame = 1

# plot(fcb_data{1}(:,4),fcb_data{1}(:,1),'.') %PE vs SSC
x = log10.(fcbData[frame][:,4])
y = log10.(fcbData[frame][:,1])

# prep color information
colo = fcbClass[frame][:]
colo[isnan.(colo)] .= 7.0
@assert sum(isnan.(colo)) == 0
colo ./= 8.0

scene = scatter(x, y, color=colo, markersize = 0.03*ones(length(x)))


## subselect two classes for development

frame = 1

msk1 = fcbClass[frame][:] .== 1
msk2 = fcbClass[frame][:] .== 6

msk = msk1 .| msk2

x = log10.(fcbData[frame][msk,4])
y1 = log10.(fcbData[frame][msk,1])
y3 = log10.(fcbData[frame][msk,3])


colo = fcbClass[frame][msk]
colo ./= 7.0

scene = scatter(x, y1, color=colo, markersize = 0.03*ones(length(x)))

scene = scatter(x, y3, color=colo, markersize = 0.03*ones(length(x)))


#
