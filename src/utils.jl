module Utils

sigmoid(x) = (1+exp(-x))^-1

pairwise(xs) = zip(xs, Iterators.drop(xs, 1))

end