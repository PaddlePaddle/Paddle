//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <time.h>
#include <fstream>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace train {

void ReadBinaryFile(const std::string& filename, std::string* contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE(static_cast<bool>(fin), "Cannot open file %s", filename);
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

std::unique_ptr<paddle::framework::ProgramDesc> Load(
    paddle::framework::Executor* executor, const std::string& model_filename) {
  VLOG(3) << "loading model from " << model_filename;
  std::string program_desc_str;
  ReadBinaryFile(model_filename, &program_desc_str);

  std::unique_ptr<paddle::framework::ProgramDesc> main_program(
      new paddle::framework::ProgramDesc(program_desc_str));
  return main_program;
}

}  // namespace train
}  // namespace paddle

int main() {
  paddle::framework::InitDevices(false);

  const auto cpu_place = paddle::platform::CPUPlace();

  paddle::framework::Executor executor(cpu_place);
  paddle::framework::Scope scope;
  auto startup_program = paddle::train::Load(&executor, "startup_program");
  auto train_program = paddle::train::Load(&executor, "main_program");

  std::string loss_name = "";
  for (auto op_desc : train_program->Block(0).AllOps()) {
    if (op_desc->Type() == "mean") {
      loss_name = op_desc->Output("Out")[0];
      break;
    }
  }

  PADDLE_ENFORCE_NE(loss_name, "", "loss not found");

  // init all parameters
  executor.Run(*startup_program, &scope, 0);

  // prepare data
  auto x_var = scope.Var("x");
  auto x_tensor = x_var->GetMutable<paddle::framework::LoDTensor>();
  x_tensor->Resize({2, 13});

  auto x_data = x_tensor->mutable_data<float>(cpu_place);
  for (int i = 0; i < 2 * 13; ++i) {
    x_data[i] = static_cast<float>(i);
  }

  auto y_var = scope.Var("y");
  auto y_tensor = y_var->GetMutable<paddle::framework::LoDTensor>();
  y_tensor->Resize({2, 1});
  auto y_data = y_tensor->mutable_data<float>(cpu_place);
  for (int i = 0; i < 2 * 1; ++i) {
    y_data[i] = static_cast<float>(i);
  }

  auto loss_var = scope.Var(loss_name);

  paddle::platform::ProfilerState pf_state;
  pf_state = paddle::platform::ProfilerState::kCPU;
  paddle::platform::EnableProfiler(pf_state);
  clock_t t1 = clock();

  for (int i = 0; i < 10; ++i) {
    executor.Run(*train_program, &scope, 0, false, true);
    std::cout << "step: " << i << " loss: "
              << loss_var->Get<paddle::framework::LoDTensor>().data<float>()[0]
              << std::endl;
  }

  clock_t t2 = clock();
  paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                    "run_paddle_op_profiler");
  std::cout << "run_time = " << t2 - t1 << std::endl;
  return 0;
}
