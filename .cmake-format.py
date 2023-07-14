# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# -----------------------------
# Options affecting formatting.
# -----------------------------
with section("format"):
    # How wide to allow formatted cmake files
    line_width = 80

# ------------------------------------------------
# Options affecting comment reflow and formatting.
# ------------------------------------------------
with section("markup"):
    # enable comment markup parsing and reflow
    enable_markup = False

    # If comment markup is enabled, don't reflow the first comment block in each
    # listfile. Use this to preserve formatting of your copyright/license
    # statements.
    first_comment_is_literal = True

# ----------------------------------
# Options affecting listfile parsing
# ----------------------------------
with section("parse"):
    # Additional FLAGS and KWARGS for custom commands
    additional_commands = {
        "cc_library": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "nv_library": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "xpu_library": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "hip_library": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "go_library": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "copy": {
            "kwargs": {
                "SRCS": '*',
                "DSTS": '*',
            }
        },
        "cc_test": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "nv_test": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "hip_test": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "xpu_test": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "go_test": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
        "py_test": {
            "kwargs": {
                "SRCS": '*',
                "DEPS": '*',
            }
        },
    }
