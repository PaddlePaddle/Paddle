# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import six
import sys
import traceback

from paddle.fluid.dygraph.dygraph_to_static.origin_info import Location, OriginInfo, global_origin_info_map

ERROR_DATA = "Error data about original source code information and traceback."

# A flag to set whether to open the dygraph2static error reporting module
SIMPLIFY_ERROR_ENV_NAME = "TRANSLATOR_SIMPLIFY_NEW_ERROR"
DEFAULT_SIMPLIFY_NEW_ERROR = 1

# A flag to set whether to display the simplified error stack
DISABLE_ERROR_ENV_NAME = "TRANSLATOR_DISABLE_NEW_ERROR"
DEFAULT_DISABLE_NEW_ERROR = 0


def attach_error_data(error, in_runtime=False):
    """
    Attachs error data about original source code information and traceback to an error.

    Args:
        error(Exception): An native error.
        in_runtime(bool): `error` is raised in runtime if in_runtime is True, otherwise in compile time
    Returns:
        An error attached data about original source code information and traceback.
    """
    e_type, e_value, e_traceback = sys.exc_info()
    tb = traceback.extract_tb(e_traceback)[1:]

    error_data = ErrorData(e_type, e_value, tb, global_origin_info_map)
    error_data.in_runtime = in_runtime

    setattr(error, ERROR_DATA, error_data)

    remove_static_file()
    return error


def remove_static_file():
    """
    Removes temporary files created during the transformation of dygraph to static graph.
    """
    del_files = set()
    for loc in global_origin_info_map:
        static_filepath = loc[0]
        del_files.add(static_filepath)

        filename, extension = os.path.splitext(static_filepath)
        del_files.add(filename + ".pyc")

    for filepath in del_files:
        if os.path.exists(filepath):
            os.remove(filepath)


class TraceBackFrame(OriginInfo):
    """
    Traceback frame information.
    """

    def __init__(self, location, function_name, source_code):
        self.location = location
        self.function_name = function_name
        self.source_code = source_code

    def formated_message(self):
        # self.source_code may be empty in some functions.
        # For example, decorator generated function
        return '    File "{}", line {}, in {}\n\t{}'.format(
            self.location.filepath, self.location.lineno, self.function_name,
            self.source_code.lstrip()
            if isinstance(self.source_code, str) else self.source_code)


class ErrorData(object):
    """
    Error data attached to an exception which is raised in un-transformed code.
    """

    def __init__(self, error_type, error_value, origin_traceback,
                 origin_info_map):
        self.error_type = error_type
        self.error_value = error_value
        self.origin_traceback = origin_traceback
        self.origin_info_map = origin_info_map
        self.in_runtime = False

    def create_exception(self):
        message = self.create_message()
        new_exception = self.error_type(message)
        setattr(new_exception, ERROR_DATA, self)
        return new_exception

    def create_message(self):
        """
        Creates a custom error message which includes trace stack with source code information of dygraph from user.
        """
        message_lines = []

        # Step1: Adds header message to prompt users that the following is the original information.
        header_message = "In transformed code:"
        message_lines.append(header_message)
        message_lines.append("")

        # Simplify error value to improve readability if error is raised in runtime
        if self.in_runtime:
            if int(
                    os.getenv(SIMPLIFY_ERROR_ENV_NAME,
                              DEFAULT_SIMPLIFY_NEW_ERROR)):
                self._simplify_error_value()
            message_lines.append(str(self.error_value))
            return '\n'.join(message_lines)

        # Step2: Optimizes stack information with source code information of dygraph from user.
        for filepath, lineno, funcname, code in self.origin_traceback:
            loc = Location(filepath, lineno)

            dygraph_func_info = self.origin_info_map.get(loc.line_location,
                                                         None)
            if dygraph_func_info:
                # TODO(liym27): more information to prompt users that this is the original information.
                # Replaces trace stack information about transformed static code with original dygraph code.
                traceback_frame = self.origin_info_map[loc.line_location]
            else:
                traceback_frame = TraceBackFrame(loc, funcname, code)

            message_lines.append(traceback_frame.formated_message())

        # Step3: Adds error message like "TypeError: dtype must be int32, but received float32".
        error_message = " " * 4 + traceback.format_exception_only(
            self.error_type, self.error_value)[0].strip("\n")
        message_lines.append(error_message)

        return '\n'.join(message_lines)

    def _simplify_error_value(self):
        """
        Simplifies error value to improve readability if error is raised in runtime.

        NOTE(liym27): The op callstack information about transformed static code has been replaced with original dygraph code.

        TODO(liym27):
            1. Need a more robust way because the code of start_trace may change.
            2. Set the switch to determine whether to simplify error_value
        """
        assert self.in_runtime is True

        error_value_lines = str(self.error_value).split("\n")
        error_value_lines_strip = [mes.lstrip(" ") for mes in error_value_lines]

        start_trace = "outputs = static_func(*inputs)"
        start_idx = error_value_lines_strip.index(start_trace)
        error_value_lines = error_value_lines[start_idx + 1:]

        error_value_str = '\n'.join(error_value_lines)
        self.error_value = self.error_type(error_value_str)

    def raise_new_exception(self):

        # Raises the origin error if disable dygraph2static error module,
        if int(os.getenv(DISABLE_ERROR_ENV_NAME, DEFAULT_DISABLE_NEW_ERROR)):
            raise

        new_exception = self.create_exception()
        if six.PY3:
            # NOTE(liym27):
            # 1. Why `raise new_exception from None`?
            #   In Python 3, by default, an new exception is raised with trace information of the caught exception.
            #   This only raises new_exception and hides unwanted implementation details from tracebacks of the
            #   caught exception.
            # 2. Use exec to bypass syntax error checking in Python 2.

            six.exec_("raise new_exception from None")
        else:
            raise new_exception
