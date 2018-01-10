"""
All util layers.
"""

from ..layer_helper import LayerHelper
from ..framework import unique_name
from ..registry import autodoc

__all__ = ['get_places']


@autodoc
def get_places(device_count=None, device_type=None):
    helper = LayerHelper('get_places', **locals())
    out_places = helper.create_variable(name=unique_name(helper.name + ".out"))
    attrs = dict()
    if device_count is not None:
        attrs['device_count'] = int(device_count)
    if device_type is not None:
        attrs['device_type'] = str(device_type)

    helper.append_op(
        type='get_places', outputs={"Out": [out_places]}, attrs=attrs)

    return out_places
