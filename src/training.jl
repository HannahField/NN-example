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
currentPath = string(basePath,digit,"/")
digits = readdir(currentPath)

chosenPicture = rand(digits)
picturePath = string(currentPath,chosenPicture)
digit -= 30
image = Gray.(FileIO.load(picturePath))

resizedImage = ImageTransformations.imresize(image,(64,64),method=ImageTransformations.Linear())

pixels = convert.(Float64,resizedImage)

network = newNetwork([64*64,100,100,10])

pixels = reshape(pixels, (64*64))
evaluation = evaluate(network,pixels)
evaluation ./= sum(evaluation)

cost = squareCost(evaluation,digit)
dump(cost)