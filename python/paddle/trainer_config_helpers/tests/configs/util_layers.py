from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

a = data_layer(name='a', size=10)
b = data_layer(name='b', size=10)

result = addto_layer(input=[a, b])
concat1 = concat_layer(input=[a, b])
concat2 = concat_layer(
    input=[identity_projection(input=a), identity_projection(input=b)])

outputs(result, concat1, concat2)
