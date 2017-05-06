from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

din = data_layer(name='data', size=30)

seq_op = [first_seq, last_seq]

agg_level = [AggregateLevel.EACH_SEQUENCE, AggregateLevel.EACH_TIMESTEP]

opts = []

for op in seq_op:
    for al in agg_level:
        opts.append(op(input=din, agg_level=al))

for op in seq_op:
    opts.append(op(input=din, agg_level=AggregateLevel.EACH_TIMESTEP, stride=5))

outputs(opts)
