/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/operators/jit/gen_base.h"
#include <fstream>
#include <iostream>
#include <sstream>

DEFINE_bool(dump_jitcode, false, "Whether to dump the jitcode to file");

namespace paddle {
namespace operators {
namespace jit {

// refer do not need useme, it would be the last one.
void GenBase::dumpCode(const unsigned char* code) const {
  if (code) {
    static int counter = 0;
    std::ostringstream filename;
    filename << "paddle_jitcode_" << name() << "." << counter << ".bin";
    counter++;
    std::ofstream fout(filename.str(), std::ios::out);
    if (fout.is_open()) {
      fout.write(reinterpret_cast<const char*>(code), this->getSize());
      fout.close();
    }
  }
}

}  // namespace jit
}  // namespace operators
}  // namespace paddle
