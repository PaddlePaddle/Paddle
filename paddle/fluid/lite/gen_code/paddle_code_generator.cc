// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include "paddle/fluid/lite/gen_code/gen_code.h"
#include "paddle/fluid/lite/model_parser/model_parser.h"

DEFINE_string(optimized_model, "", "");
DEFINE_string(generated_code_file, "__generated_code__.cc", "");

namespace paddle {
namespace lite {
namespace gencode {

void GenCode(const std::string& model_dir, const std::string& out_file) {
  lite::Scope scope;
  framework::proto::ProgramDesc desc;
  LoadModel(model_dir, &scope, &desc);

  ProgramCodeGenerator codegen(desc, scope);

  std::ofstream file(out_file);

  file << codegen.GenCode();

  file.close();
}

}  // namespace gencode
}  // namespace lite
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, false);
  paddle::lite::gencode::GenCode(FLAGS_optimized_model,
                                 FLAGS_generated_code_file);
  return 0;
}
