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


#  str and bytes related functions
def to_text(obj, encoding='utf-8', inplace=False):
    """
    All string in PaddlePaddle should be represented as a literal string.

    This function will convert object to a literal string without any encoding.
    Especially, if the object type is a list or set container, we will iterate
    all items in the object and convert them to literal string.

    In Python3:
        Decode the bytes type object to str type with specific encoding

    In Python2:
        Decode the str type object to unicode type with specific encoding

    Args:
        obj(unicode|str|bytes|list|set) : The object to be decoded.
        encoding(str) : The encoding format to decode a string
        inplace(bool) : If we change the original object or we create a new one

    Returns:
        Decoded result of obj

    Examples:

        .. code-block:: python

            import paddle

            data = "paddlepaddle"
            data = paddle.compat.to_text(data)
            # paddlepaddle

    """
    if obj is None:
        return obj

    if isinstance(obj, list):
        if inplace:
            for i in six.moves.xrange(len(obj)):
                obj[i] = _to_text(obj[i], encoding)
            return obj
        else:
            return [_to_text(item, encoding) for item in obj]
    elif isinstance(obj, set):
        if inplace:
            for item in obj:
                obj.remove(item)
                obj.add(_to_text(item, encoding))
            return obj
        else:
            return set([_to_text(item, encoding) for item in obj])
    elif isinstance(obj, dict):
        if inplace:
            new_obj = {}
            for key, value in six.iteritems(obj):
                new_obj[_to_text(key, encoding)] = _to_text(value, encoding)
            obj.update(new_obj)
            return obj
        else:
            new_obj = {}
            for key, value in six.iteritems(obj):
                new_obj[_to_text(key, encoding)] = _to_text(value, encoding)
            return new_obj
    else:
        return _to_text(obj, encoding)


def _to_text(obj, encoding):
    """
    In Python3:
        Decode the bytes type object to str type with specific encoding

    In Python2:
        Decode the str type object to unicode type with specific encoding,
        or we just return the unicode string of object

    Args:
        obj(unicode|str|bytes) : The object to be decoded.
        encoding(str) : The encoding format

    Returns:
        decoded result of obj
    """
    if obj is None:
        return obj

    if isinstance(obj, six.binary_type):
        return obj.decode(encoding)
    elif isinstance(obj, six.text_type):
        return obj
    elif isinstance(obj, (bool, float)):
        return obj
    else:
        return six.u(obj)
