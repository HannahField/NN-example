mutable struct Layer
    weights::Matrix{Float32}
    biases::Vector{Float32}
    previous::Union{Layer,Nothing}
    next::Union{Layer,Nothing}
end

function newLayer(height,width,previous::Layer) 
        layer = Layer(    
        (rand(Float32, (width, height)) .- 0.5) .* 2,
        (rand(Float32, width) .- 0.5) .* 2,
        previous,
        nothing
        )
    previous.next = layer
    layer
end




newLayer(height,width) = Layer(
    (rand(Float32, (width, height)) .- 0.5) .* 2,
    (rand(Float32, width) .- 0.5) .* 2,
    nothing,
    nothing
)


newLayer(height,width,previous::Nothing) = Layer(
    (rand(Float32, (width, height)) .- 0.5) .* 2,
    (rand(Float32, width) .- 0.5) .* 2,
    previous,
    nothing
)

