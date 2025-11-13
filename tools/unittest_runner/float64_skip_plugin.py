# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
float64_skip_plugin.py

pytest plugin to skip test cases that use float64 data type.
When the environment variable FLAG_SKIP_FLOAT64 is set to 1, this plugin will skip all test cases that use float64.
"""

import inspect
import os
import traceback

import numpy as np
import pytest


def pytest_collection_modifyitems(config, items):
    """
    Skip tests whose class or base classes (up to but NOT including OpTest) contain
    'float64' in method source or class attributes.
    """
    skip_float64 = os.environ.get("FLAG_SKIP_FLOAT64", "0") == "1"
    if not skip_float64:
        return

    for item in items:
        try:
            test_class = getattr(item, "cls", None)
            if test_class is None:
                continue

            skip_test = False
            debug_info = None

            # Walk MRO from the class itself upward, but STOP when we reach OpTest or object.
            for cls in inspect.getmro(test_class):
                # Stop traversing when we reach OpTest or object (do not inspect OpTest or above)
                if cls is object or cls.__name__ == "OpTest":
                    break

                # Only iterate attributes actually defined on this class (avoid inherited ones)
                for name, val in cls.__dict__.items():
                    if name.startswith("_"):
                        continue

                    # 1) If attribute is a function/staticmethod/classmethod/property -> extract the underlying func
                    func = None
                    if isinstance(val, staticmethod):
                        func = val.__func__
                    elif isinstance(val, classmethod):
                        func = val.__func__
                    elif isinstance(val, property):
                        # check fget/fset/fdel if present
                        for f in (val.fget, val.fset, val.fdel):
                            if f is not None:
                                try:
                                    src = inspect.getsource(f)
                                except Exception:
                                    src = ""
                                if "float64" in src:
                                    skip_test = True
                                    debug_info = (
                                        cls,
                                        name,
                                        "property method",
                                        f,
                                    )
                                    break
                        if skip_test:
                            break
                        continue
                    elif callable(val):
                        func = val

                    # If we have a function, check its source code
                    if func is not None:
                        try:
                            src = inspect.getsource(func)
                            if "float64" in src:
                                skip_test = True
                                debug_info = (cls, name, "callable", func)
                                break
                        except (OSError, TypeError):
                            # source not available (e.g., builtins, C-extensions) — ignore
                            pass
                        except Exception:
                            # safeguard: don't crash plugin
                            pass
                    else:
                        # 2) Non-callable attribute: check if its string contains float64 or it is np.float64/dtype
                        try:
                            # direct dtype object check
                            if val is np.float64:
                                skip_test = True
                                debug_info = (
                                    cls,
                                    name,
                                    "np.float64 object",
                                    val,
                                )
                                break
                            if isinstance(val, np.dtype) and val == np.dtype(
                                "float64"
                            ):
                                skip_test = True
                                debug_info = (
                                    cls,
                                    name,
                                    "np.dtype('float64')",
                                    val,
                                )
                                break
                            # fallback: string representation check (handles things like "float64" or "np.float64")
                            if "float64" in repr(val) or "float64" in str(val):
                                skip_test = True
                                debug_info = (
                                    cls,
                                    name,
                                    "attr repr/str contains float64",
                                    val,
                                )
                                break
                        except Exception:
                            # ignore weird attributes that raise on repr
                            pass

                if skip_test:
                    break

            if skip_test:
                # debug print - helps find exactly which class/attr triggered the skip
                try:
                    cls, name, kind, what = debug_info
                    print(
                        f"[SKIP-FLOAT64] Skipping test {item.nodeid}: detected 'float64' in {kind} "
                        f"'{name}' of class {cls.__module__}.{cls.__name__}. (repr: {what!r})"
                    )
                except Exception:
                    print(
                        f"[SKIP-FLOAT64] Skipping test {item.nodeid}: detected 'float64' (debug info unavailable)"
                    )
                item.add_marker(pytest.mark.skip(reason="SKIP FLOAT64 TESTS"))

        except Exception:
            # don't let a plugin crash the collection
            traceback.print_exc()
            continue
