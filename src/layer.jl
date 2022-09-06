struct Layer
    weights::Matrix{Float32}
    biases::Vector{Float32}
end

newLayer(height,width) = Layer(
    (rand(Float32, (width, height)) .- 0.5) .* 2,
    (rand(Float32, width) .- 0.5) .* 2
)