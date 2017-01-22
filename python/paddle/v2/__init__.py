"""
This is an experimental package for Paddle new API.

Currently, we use should always use

..  code-block: python

    import paddle.v2 as paddle

as our import statement. The API is in flux, never use this package in
production.
"""

import optimizer

__all__ = ['optimizer']
