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


class BaseTranspiler:
    '''
    Base class for transpilers.
    '''

    def _remove_unused_var(self):
        '''
        remove unused varibles in program
        '''
        args = []
        for i in range(len(self.block.ops)):
            current_op = self.block.ops[i]
            args += current_op.input_arg_names
            args += current_op.output_arg_names
        args = list(set(args))  # unique the input and output arguments

        for var in self.block.vars.keys():
            if var not in args:
                self.block._remove_var(var)
