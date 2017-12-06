from paddle.trainer_config_helpers import *

vec1 = data_layer(name='vector1', size=10)
vec2 = data_layer(name='vector2', size=10)
dot_product = dot_prod_layer(input1=vec1, input2=vec2)

outputs(dot_product)
