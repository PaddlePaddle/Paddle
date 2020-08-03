# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import warnings
import functools
import inspect

def deprecated(since="", new_api="", detail=""):
    """Decorate a function to signify its deprecation.
       
       This function wraps a method that will soon be removed and does two things:
           - The docstring of the API will be modified to include a notice
             about deprecation."
           - Raises a :class:`~deprecation.DeprecatedWarning` when old API is called.
       :param since: The version at which the decorated method is considered deprecated.
       :param new_api: The new API users should use.
       :param detail: Extra details about the deprecation. For example, the reason why the API is deprecated.
    """
    def decorator(func):
        assert isinstance(since, str)
        assert isinstance(new_api, str)
        assert isinstance(detail, str)

        since_ = since.strip()
        new_api_ = new_api.strip()
        detail_ = detail.strip()

        assert new_api_.startswith("paddle."), 'new_api must start with "paddle."'
        assert len(new_api_) > len("paddle."), "no api found in new_api."

        msg = "API {}.{} is deprecated".format(func.__module__, func.__name__)
        if len(since_) > 0:
            msg += " since {}".format(since_)
        msg += "."
        msg = msg + " Use {} instead.".format(new_api_)
        if len(detail_) > 0:
            msg += "\n Detail: {}".format(detail_)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning) # reset filter
            return func(*args, **kwargs)

        return wrapper
    return decorator
    
