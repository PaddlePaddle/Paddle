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
"""
decorator to deprecate a function or class
"""

import functools
import sys
import warnings

import paddle

__all__ = []


def deprecated(update_to="", since="", reason="", level=0):
    """Decorate a function to signify its deprecation.

    This function wraps a method that will soon be removed and does two things:
        - The docstring of the API will be modified to include a notice
          about deprecation."
        - Raises a :class:`~exceptions.DeprecatedWarning` when old API is called.

    Args:
         since(str, optional): The version at which the decorated method is considered deprecated.
         update_to(str, optional): The new API users should use.
         reason(str, optional): The reason why the API is deprecated.
         level(int, optional): The deprecated warning log level. It must be
             an Integer and must be one of 0, 1, 2.
             If `level == 0`, the warning message will not be showed.
             If `level == 1`, the warning message will be showed normally.
             If `level == 2`, it will raise `RuntimeError`.

    Returns:
        decorator: decorated function or class.
    """

    def decorator(func):
        # TODO(zhiqiu): temporally disable the warnings
        return func
        """construct warning message, and return a decorated function or class."""
        assert isinstance(update_to, str), 'type of "update_to" must be str.'
        assert isinstance(since, str), 'type of "since" must be str.'
        assert isinstance(reason, str), 'type of "reason" must be str.'
        assert isinstance(level, int) and level >= 0 and level < 3, (
            'type of "level" must be int and must be one of 0, 1, 2. But '
            'received: {}.'.format(level)
        )

        _since = since.strip()
        _update_to = update_to.strip()
        _reason = reason.strip()

        msg = f'API "{func.__module__}.{func.__name__}" is deprecated'

        if len(_since) > 0:
            msg += f" since {_since}"
        msg += ", and will be removed in future versions."
        if len(_update_to) > 0:
            assert _update_to.startswith(
                "paddle."
            ), 'Argument update_to must start with "paddle.", your value is "{}"'.format(
                update_to
            )
            msg += f' Please use "{_update_to}" instead.'
        if len(_reason) > 0:
            msg += f"\nreason: {_reason}"
        if func.__doc__:
            func.__doc__ = ('\n\nWarning: ' + msg + '\n') + func.__doc__

        if level == 0:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """deprecated warning should be fired in 3 circumstances:
            1. current version is develop version, i.e. "0.0.0", because we assume develop version is always the latest version.
            2. since version is empty, in this case, API is deprecated in all versions.
            3. current version is newer than since version.
            """

            if level == 2:
                raise RuntimeError(
                    'API "{}.{}" has been deprecated.'.format(
                        func.__module__, func.__name__
                    )
                )

            warningmsg = "\033[93m\nWarning:\n%s \033[0m" % (msg)
            # ensure ANSI escape sequences print correctly in cmd and powershell
            if sys.platform.lower() == 'win32':
                warningmsg = "\nWarning:\n%s " % (msg)

            v_current = [int(i) for i in paddle.__version__.split(".")]
            v_current += [0] * (4 - len(v_current))
            v_since = [int(i) for i in _since.split(".")]
            v_since += [0] * (4 - len(v_since))
            if (
                paddle.__version__ == "0.0.0"
                or _since == ""
                or v_current >= v_since
            ):
                warnings.warn(
                    warningmsg, category=DeprecationWarning, stacklevel=2
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator
