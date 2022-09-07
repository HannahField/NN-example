module Utils

sigmoid(x) = (1+exp(-x))^-1

sigmoidDif(x) = exp(x)/(exp(x)+1)^2

pairwise(xs) = zip(xs, Iterators.drop(xs, 1))

end