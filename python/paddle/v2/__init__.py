"""
This is an experimental package for Paddle new API.

Currently, we use should always use

..  code-block: python

    import paddle.v2 as paddle

as our import statement. The API is in flux, never use this package in
production.
"""

import py_paddle.swig_paddle as raw
import config
import data
import paddle.proto as proto
import layers
import optimizer
import model

__all__ = ['config', 'data', 'raw', 'proto', 'layers', 'optimizer', 'model']
