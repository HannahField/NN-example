include("utils.jl")
include("layer.jl")
include("network.jl")

network = newNetwork([2,3,3,2])

evaluate(network, (rand(Float64,2).-0.5)*2)