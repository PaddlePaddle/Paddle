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

from paddle.fluid.dygraph.dygraph_to_static.origin_info import Location, OriginInfo, ORIGI_INFO_MAP, global_origin_info_map
import traceback, sys

ERROR_DATA = "Error data about original source code information and traceback."


def attach_error_data(error):
    """
    Attachs error data about original source code information and traceback to an error.

    Args:
        error(Exception): An native error.

    Returns:
        An error attached data about original source code information and traceback.
    """
    e_type, e_value, tb = sys.exc_info()
    tb = traceback.extract_tb(tb)[1:]

    error_data = ErrorData(e_type, e_value, tb, global_origin_info_map)
    setattr(error, ERROR_DATA, error_data)

    return error


class TraceBackFrame(OriginInfo):
    """
    Traceback frame information.
    """

    def __init__(self, location, function_name, source_code):
        self.location = location
        self.function_name = function_name
        self.source_code = source_code


class ErrorData(object):
    """
    Error data attached to an exception which is raised in un-transformed code.

    TODO(liym27): Consider the case that op_callstack when error raised from c++ code
    """

    def __init__(self, error_type, error_value, origin_tb, origin_info_map):
        self.error_type = error_type
        self.error_value = error_value
        self.origin_tb = origin_tb
        self.origin_info_map = origin_info_map

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
        header_message = "In user code:"
        message_lines.append(header_message)
        message_lines.append("")

        # Step2: Optimizes stack information with source code information of dygraph from user.
        for filepath, lineno, funcname, code in self.origin_tb:
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
