from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=256)
glu = gated_unit_layer(
    size=512,
    input=data,
    act=TanhActivation(),
    gate_attr=ExtraLayerAttribute(error_clipping_threshold=100.0),
    gate_param_attr=ParamAttr(initial_std=1e-4),
    gate_bias_attr=ParamAttr(initial_std=1),
    inproj_attr=ExtraLayerAttribute(error_clipping_threshold=100.0),
    inproj_param_attr=ParamAttr(initial_std=1e-4),
    inproj_bias_attr=ParamAttr(initial_std=1),
    layer_attr=ExtraLayerAttribute(error_clipping_threshold=100.0))

outputs(glu)
