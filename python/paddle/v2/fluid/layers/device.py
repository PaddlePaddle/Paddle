"""
All util layers.
"""

from ..layer_helper import LayerHelper
from ..framework import unique_name

__all__ = ['get_places']


def get_places(device_count=0, device_type="CPU"):
    helper = LayerHelper('get_places', **locals())
    out_places = helper.create_variable(name=unique_name(helper.name + ".out"))
    helper.append_op(
        type='get_places',
        outputs={"Out": [out_places]},
        attrs={
            "device_type": device_type,
            'device_count': device_count,
        })

    return out_places
