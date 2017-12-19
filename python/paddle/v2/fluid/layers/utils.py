"""
All util layers.
"""

from ..layer_helper import LayerHelper
from ..framework import Variable

__all__ = ['get_places']


def get_places(trainer_count, device_type="CPU"):
    helper = LayerHelper('get_places', **locals())
    out_places = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='get_places',
        outputs={"Out": [out_places]},
        attrs={
            "device_type": device_type,
            'trainer_count': trainer_count,
        })

    return out_places
