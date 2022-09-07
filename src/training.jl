include("utils.jl")
include("layer.jl")
include("network.jl")
include("cost.jl")
using Images
using FileIO

#TRAINING DATA IS THE NIST SPECIAL DATABASE 19. (https://www.nist.gov/srd/nist-special-database-19). DOWNLOAD THE BY_FIELD.ZIP ONE. WE WILL ONLY USE THE DIGITS IN THIS PROJECT,
# BUT YOU CAN EASILY EXPAND TO BE UPPER AND LOWER CASE LETTERS TOO.

basePath = "./Data/T0/"
digit = rand(30:39)
currentPath = string(basePath, digit, "/")
digits = readdir(currentPath)

chosenPicture = rand(digits)
picturePath = string(currentPath, chosenPicture)
digit -= 30
image = Gray.(FileIO.load(picturePath))

resizedImage = ImageTransformations.imresize(image, (64, 64), method=ImageTransformations.Linear())

pixels = convert.(Float64, resizedImage)

network = newNetwork([64 * 64, 100, 100, 10])

pixels = reshape(pixels, (64 * 64))
evaluation = evaluate(network, pixels)
evaluation ./= sum(evaluation)

expectedOutput = zeros(10)
expectedOutput[digit+1] = 1

cost = squareCost(evaluation, expectedOutput)

dA_dAPrev(layer, input) = transpose(Utils.sigmoidDif.(layer.weights * evaluate(layer.previous,input)+layer.biases))*layer.weights
a = network.weights
b = Utils.sigmoidDif.(network.weights*evaluate(network.previous,pixels)+network.biases)
dA_dw(layer, input) = evaluate(layer.previous, input) *
                      transpose(Utils.sigmoidDif.(
    layer.weights * evaluate(layer.previous, input) + layer.biases)
)
dA_db(layer, input) = transpose(Utils.sigmoidDif.(
    layer.weights * evaluate(layer.previous, input) + layer.biases)
)


function dC_dA(layer::Union{Layer,Nothing}, input, expected)
    if (layer.next === nothing)
        2 .* (evaluate(layer, input) .- expected)
    else
        sum((dC_dA(layer.next, input, expected)) .* dA_dAPrev(layer.next,input),dims=1)
    end
end

dC_dw(layer, input, expected) = transpose(dA_dw(layer, input)).*(dC_dA(layer, input, expected))
dC_db(layer, input, expected) = transpose(dA_db(layer, input)).*(dC_dA(layer, input, expected))

#dump("dA_db")
#dump(dA_db(network,pixels))
#dump("dA_dw")
#dump(dA_dw(network,pixels))
#dump("dC_dA")
#dump(dC_dA(network, pixels, expectedOutput))
#dump("dC_dw")
#dump(dC_dw(network, pixels, expectedOutput))
#dump("dC_db")
#dump(dC_db(network, pixels, expectedOutput))