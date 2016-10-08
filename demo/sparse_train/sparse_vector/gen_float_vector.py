from random import uniform,randint

# total samples to generate
samples=2000

# total DIMENSION
dimension=10000000

# float value number
float_num_min=10
float_num_max=100

# float value range
float_min=0.01
float_max=5.00

# output range
regression_min=0.01
regression_max=100.00

data = "train.txt"


'''
generate dummy data for testing sparse_float_vector input
format:
    label, [index, value], [index, value], ....
'''
with open(data, "w") as f:
    for ids in range(samples):
        num = randint(float_num_min, float_num_max)
        index = 0
        value = 0.0
        output = uniform(regression_min, regression_max)
        f.write("%f," % output)
        for i in range(num):
            index = randint(0, dimension)
            value = uniform(float_min, float_max)
            if i == num -1:
                f.write("%d %f" % (index, value))
            else:
                f.write("%d %f," % (index, value))

        f.write("\n")


