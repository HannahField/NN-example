include("layer.jl")
include("utils.jl")

activation(nodeValue) = Utils.sigmoid(nodeValue)
activationDif(nodeValue) = Utils.sigmoidDif(nodeValue)

#newNetwork(size::Vector{Int64}) = (
#    map(
#        (x) -> newLayer(x...),
#        Utils.pairwise(size)
#    )
#)

newNetwork(size::Vector{Int64}) = foldl(
        (acc,x) -> newLayer(x...,acc),
        Utils.pairwise(size),
        init=nothing
)

evaluate(layer::Layer, input::Vector{Float64}) = activation.(acc.weights * evaluate(layer.previous,input) + acc.biases)

evaluate(layer::Nothing, input::Vector{Float64}) = input


stepEvaluate(layer::Layer, input::Vector{Float64}) = layer.weights * input + layer.biases

