include("utils.jl")
include("layer.jl")
include("network.jl")
using Images
using FileIO


basePath = "./Data/T0/"
digit = rand(30:39)
currentPath = string(basePath,digit,"/")
digits = readdir(currentPath)

chosenPicture = rand(digits)
picturePath = string(currentPath,chosenPicture)

image = Gray.(FileIO.load(picturePath))

resizedImage = ImageTransformations.imresize(image,(64,64),method=ImageTransformations.Linear())

pixels = convert.(Float64,resizedImage)

network = newNetwork([64*64,100,100,10])

pixels = reshape(pixels, (64*64))
dump(chosenPicture)
dump(evaluate(network,pixels))