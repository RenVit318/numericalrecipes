#################
#
# Code related to probability distributions
# 
#################

def rejection_sampling(func, rng, N,
                       shift_x=lambda x: x,
                       shift_y=lambda x: x,
                       x0 = 4891653):
    """Sample a 1D distribution using rejection sampling. This function generates
    two random numbers using the provided rng. It then assigns the first value as
    'x' and shifts it using the shift_x function, and assigns the second value as
    'y' and shifts it using the shift_y function. If y<func(x) the point is accepted
    Repeat this until we have N points, and return these
    x0 is used as a starting seed for the rng
    """

    sampled_points = np.zeros(N)
    num_tries = 0 # For performance testing
    for i in range(N):
        not_sampled = True

        # Keep sampling until we find a x,y pair that fits
        while not_sampled:
            numbers, x0 = rng(2, x0=x0, return_laststate=True) # This is now U(0,1)

            x = shift_x(numbers[0])
            y = shift_y(numbers[1])
            num_tries += 1
            
            if y < func(x):
                sampled_points[i] = x
                not_sampled = False

    print(f'Average No. tries: {num_tries/N:.1f}')
    return sampled_points
