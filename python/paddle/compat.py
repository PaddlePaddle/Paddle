#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import six

__all__ = []


def to_bytes(obj, encoding='utf-8', inplace=False):
    """
    All string in PaddlePaddle should be represented as a literal string.

    This function will convert object to a bytes with specific encoding.
    Especially, if the object type is a list or set container, we will iterate
    all items in the object and convert them to bytes.

    In Python3:
        Encode the str type object to bytes type with specific encoding

    In Python2:
        Encode the unicode type object to str type with specific encoding,
        or we just return the 8-bit string of object

    Args:
        obj(unicode|str|bytes|list|set) : The object to be encoded.
        encoding(str) : The encoding format to encode a string
        inplace(bool) : If we change the original object or we create a new one

    Returns:
        Decoded result of obj

    Examples:

        .. code-block:: python

            import paddle

            data = "paddlepaddle"
            data = paddle.compat.to_bytes(data)
            # b'paddlepaddle'

    """
    if obj is None:
        return obj

    if isinstance(obj, list):
        if inplace:
            for i in six.moves.xrange(len(obj)):
                obj[i] = _to_bytes(obj[i], encoding)
            return obj
        else:
            return [_to_bytes(item, encoding) for item in obj]
    elif isinstance(obj, set):
        if inplace:
            for item in obj:
                obj.remove(item)
                obj.add(_to_bytes(item, encoding))
            return obj
        else:
            return set([_to_bytes(item, encoding) for item in obj])
    else:
        return _to_bytes(obj, encoding)


def _to_bytes(obj, encoding):
    """
    In Python3:
        Encode the str type object to bytes type with specific encoding

    In Python2:
        Encode the unicode type object to str type with specific encoding,
        or we just return the 8-bit string of object

    Args:
        obj(unicode|str|bytes) : The object to be encoded.
        encoding(str) : The encoding format

    Returns:
        encoded result of obj
    """
    if obj is None:
        return obj

    assert encoding is not None
    if isinstance(obj, six.text_type):
        return obj.encode(encoding)
    elif isinstance(obj, six.binary_type):
        return obj
    else:
        return six.b(obj)
