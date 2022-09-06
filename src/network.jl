struct Network
    layers::Vector{Layer}
end

activation(nodeValue) = Utils.sigmoid(nodeValue)

newNetwork(size::Vector{Int64}) = Network(
    map(
        (x) -> newLayer(x...),
        Utils.pairwise(size)
    )
)

evaluate(network::Network, input::Vector{Float64}) = reduce(
    (x, acc) -> activation.(acc.weights * x + acc.biases),
    network.layers,
    init=input
)
