from .. import core
from ..layer_helper import LayerHelper

__all__ = ['data']


def data(name,
         shape,
         append_batch_size=True,
         dtype='float32',
         lod_level=0,
         type=core.VarDesc.VarType.LOD_TENSOR,
         main_program=None,
         startup_program=None,
         stop_gradient=True):
    """
    Data Layer.

    Args:
       name: The name/alias of the function
       shape: Tuple declaring the shape.
       append_batch_size: Whether or not to append the data as a batch.
       dtype: The type of data : float32, float_16, int etc
       type: The output type. By default it is LOD_TENSOR.
       lod_level(int): The LoD Level. 0 means the input data is not a sequence.
       main_program: Name of the main program that calls this
       startup_program: Name of the startup program
       stop_gradient: A boolean that mentions whether gradient should flow.

    This function takes in input and based on whether data has
    to be returned back as a minibatch, it creates the global variable using
    the helper functions. The global variables can be accessed by all the
    following operations and layers in the graph.

    All the input variables of this function are passed in as local variables
    to the LayerHelper constructor.

    """
    helper = LayerHelper('data', **locals())
    shape = list(shape)
    for i in xrange(len(shape)):
        if shape[i] is None:
            shape[i] = -1
            append_batch_size = False
        elif shape[i] < 0:
            append_batch_size = False

    if append_batch_size:
        shape = [-1] + shape  # append batch size as -1

    return helper.create_global_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=type,
        stop_gradient=stop_gradient,
        lod_level=lod_level)
