#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Function for augment error message hint for users."""

from __future__ import print_function

import collections
import itertools
import os
import re
import six
from .. import compat as cpt

# Names for indices into traceback tuples.
TB_FILENAME = 0
TB_LINENO = 1

_NAME_REGEX = r"[A-Za-z0-9.][A-Za-z0-9_.\-/]*?"
_TAG_REGEX = r"{{{{({name}) ({name})}}}}".format(name=_NAME_REGEX)
_TARGET_REGEX = r"^(.*?)({tag})".format(tag=_TAG_REGEX)
_TARGET_PATTERN = re.compile(_TARGET_REGEX, re.DOTALL)

# Type field can retain extensibility
_ParseTag = collections.namedtuple("_ParseTag", ["type", "name"])

# Used to filter out the library path
_BAD_PATH_SUBSTRINGS = [
    os.path.join("paddle", "fluid"),
    os.path.join("paddle", "distributed"),
    os.path.join("paddle", "reader"),
    os.path.join("paddle", "utils"),
    "<embedded",
]


def _parse_target_tuple(message):
    """
    Parse the tuple message we want augment from message.

    Splits the message into separators and tags. Tags are named tuples
    representing the string {{type name}} and they are separated by 
    separators.

    For example, the message "123{{operator mul}}456" after being processed
    is like seps = ["123", "456"], tags = [_ParseTag("node", "Foo")].

    Args:
        messsge(String): String to parse

    Returns:
        list: (list of separator strings, list of _ParseTags)
    """
    seps = []
    tags = []
    pos = 0
    while pos < len(message):
        match = re.match(_TARGET_PATTERN, message[pos:])
        if match:
            seps.append(match.group(1))
            tags.append(_ParseTag(match.group(3), match.group(4)))
            pos += match.end()
        else:
            break
    seps.append(message[pos:])
    return seps, tags


def _get_defining_frame_of_op(error_traceback):
    """
    Find and return the file name and line number where op was defined.

    Args:
        error_traceback(String): exception info return from core.
    
    Returns:
        filename(String|None): the file name where op was defined.
        lineno(String|None): the line number where op wa defined.
    """
    for frame in reversed(error_traceback.splitlines()):
        frame = frame.strip()
        if frame.startswith('File'):
            frame_pieces = frame.split(',')
            filename = frame_pieces[TB_FILENAME].strip().split(' ')[1].strip(
                '"')
            lineno = frame_pieces[TB_LINENO].strip().split(' ')[1]
            contains_bad_substrs = [
                ss in filename for ss in _BAD_PATH_SUBSTRINGS
            ]
            if not any(contains_bad_substrs):
                return filename, lineno
    return None, None


def _hint_augment(error_traceback):
    """
    Augment the operator execution error hint in error message.
    
    The [[{{operator <op_type>}}]] in error message will be replaced 
    by [[operator <op_type> execution error (defined at <file_path>:
    <line_num>)]].

    Args:
        error_traceback(String): exception info return from core.

    Returns:
        String: the error traceback be augmented.
    """
    seps, tags = _parse_target_tuple(error_traceback)
    filename, lineno = _get_defining_frame_of_op(error_traceback)
    subs = []

    for t in tags:
        msg = "%s %s error" % (t.type, t.name)
        if t.type == "operator" and filename is not None and lineno is not None:
            msg = "operator %s execution error (defined at %s:%s)" % (
                t.name, filename, lineno)
        subs.append(msg)

    return "".join(
        itertools.chain(
            *six.moves.zip_longest(
                seps, subs, fillvalue="")))


def paddle_enforce_handler(ex_type, ex_val, ex_traceback):
    """
    Define the core.EnforceNotMet exception handler to shape the error message stack.
    This Inferface needs to be used with the sys.excepthook.

    Args:
        ex_type(exception type): the exception type of the exception being handled.
        ex_val(exception value): the exception parameter (its associated value or the second argument to raise).
        ex_traceback(exception traceback): a traceback object which encapsulates the call stack at the point where the exception originally occurred.

    Returns: None

    Examples:
        .. code-block:: python

        import sys
        import paddle.fluid.core as core
        from paddle.fluid import error_format

        try:
            core.__unittest_throw_exception__()
        except Exception as e:
            if isinstance(e, core.EnforceNotMet):
                sys.excepthook = error_format.paddle_enforce_handler
            six.reraise(*sys.exc_info())

    """
    ex_msg_list = _hint_augment(str(cpt.to_text(ex_val))).split(
        "PaddleEnforceError.")
    print(ex_msg_list[0])
    if len(ex_msg_list) > 0:
        print(
            "----------------------\nError Message Summary:\n----------------------\nError:%s"
            % ex_msg_list[1])
