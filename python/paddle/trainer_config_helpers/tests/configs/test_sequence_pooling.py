from paddle.trainer_config_helpers import *

settings(learning_rate=1e-4, batch_size=1000)

din = data_layer(name='dat_in', size=100)

POOL_TYPE = [MaxPooling, AvgPooling, SumPooling]

AGG_LEVEL = [AggregateLevel.TO_SEQUENCE, AggregateLevel.TO_NO_SEQUENCE]

opts = []

for pt in POOL_TYPE:
    for al in AGG_LEVEL:
        opts.append(pooling_layer(input=din, agg_level=al, pooling_type=pt()))

for pt in POOL_TYPE:
    opts.append(
        pooling_layer(
            input=din,
            agg_level=AggregateLevel.TO_NO_SEQUENCE,
            pooling_type=pt(),
            stride=5))

opts.append(
    pooling_layer(
        input=din, pooling_type=MaxPooling(output_max_index=True)))

outputs(opts)
