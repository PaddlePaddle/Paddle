'''
Test mixed layer, projections and operators.
'''
from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-4)

din = data_layer(name='test', size=100)

din = embedding_layer(input=din, size=256)

with mixed_layer(size=100) as m1:
    m1 += full_matrix_projection(input=din)

with mixed_layer(size=100) as m2:
    m2 += table_projection(input=m1)

with mixed_layer(size=100) as m3:
    m3 += identity_projection(input=m2)

with mixed_layer(size=100) as m4:
    m4 += dotmul_projection(input=m3)

with mixed_layer() as m5:
    m5 += context_projection(input=m4, context_len=3)

with mixed_layer() as m6:
    m6 += dotmul_operator(a=m3, b=m4)
    m6 += scaling_projection(m3)

img = data_layer(name='img', size=32 * 32)
flt = data_layer(name='filter', size=3 * 3 * 1 * 64)

with mixed_layer() as m7:
    m7 += conv_operator(
        img=img, filter=flt, num_filters=64, num_channels=1, filter_size=3)

end = mixed_layer(
    input=[
        full_matrix_projection(input=m5),
        trans_full_matrix_projection(input=m6), full_matrix_projection(input=m7)
    ],
    size=100,
    layer_attr=ExtraAttr(
        drop_rate=0.5, error_clipping_threshold=40))

outputs(end)
