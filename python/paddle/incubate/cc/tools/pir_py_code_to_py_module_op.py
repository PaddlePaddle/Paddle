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

from absl import app, flags

from paddle.incubate.cc.py_code_gen.tools.pir_py_code_to_py_module_op import (
    TranslatePirPyCodeToPyModuleOp,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("ir_program", "", "ir programs file.")
flags.DEFINE_string("output", "", "ir programs file.")


def main(argv):
    pycode = TranslatePirPyCodeToPyModuleOp(FLAGS.ir_program)
    WriteToFile(FLAGS.output, pycode)


def WriteToFile(filepath, py_code):
    with open(filepath, "w") as f:
        f.write(py_code)


if __name__ == "__main__":
    app.run(main)
