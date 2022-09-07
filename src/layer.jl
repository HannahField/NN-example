struct Layer
    weights::Matrix{Float32}
    biases::Vector{Float32}
    previous::Union{Layer,Nothing}
end

newLayer(height,width,previous::Layer) = Layer(
    (rand(Float32, (width, height)) .- 0.5) .* 2,
    (rand(Float32, width) .- 0.5) .* 2,
    previous
)

newLayer(height,width) = Layer(
    (rand(Float32, (width, height)) .- 0.5) .* 2,
    (rand(Float32, width) .- 0.5) .* 2,
    nothing
)


newLayer(height,width,previous::Nothing) = Layer(
    (rand(Float32, (width, height)) .- 0.5) .* 2,
    (rand(Float32, width) .- 0.5) .* 2,
    previous
)
