def mse(actual, predicted):
    return (actual-predicted)**2

def mse_prime(actual, predicted):
    return (2)*(predicted-actual)
