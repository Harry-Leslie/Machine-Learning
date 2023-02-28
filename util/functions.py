def commuteMLL(weights, inputs, bias):
    s = []
    for i in range(len(weights)):
        s.append(weights[i]*inputs[i])
    s.append(bias)
    return sum(s)

