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

#include <sstream>
#include <string>
#include <vector>
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/api/paddle_use_kernels.h"
#include "paddle/fluid/lite/api/paddle_use_ops.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/tools/debug/debug_utils.h"

namespace paddle {
namespace lite {
namespace tools {
namespace debug {

void Run(DebugConfig* conf) {
  CHECK(conf);
#ifdef LITE_WITH_ARM
  DeviceInfo::Init();
  DeviceInfo::Global().SetRunMode(LITE_POWER_HIGH, conf->arm_thread_num);
#endif
  lite::Predictor predictor;
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
#ifdef LITE_WITH_ARM
      Place{TARGET(kARM), PRECISION(kFloat)},
#endif
#ifdef LITE_WITH_X86
      Place{TARGET(kX86), PRECISION(kFloat)},
#endif
  });

  std::vector<std::string> passes{{
      "static_kernel_pick_pass", "variable_place_inference_pass",
      "type_target_cast_pass", "variable_place_inference_pass",
      "io_copy_kernel_pick_pass", "variable_place_inference_pass",
      "runtime_context_assign_pass",
  }};

  predictor.Build(conf->model_dir,
#ifdef LITE_WITH_ARM
                  Place{TARGET(kARM), PRECISION(kFloat)},
#endif
#ifdef LITE_WITH_X86
                  Place{TARGET(kX86), PRECISION(kFloat)},
#endif
                  valid_places, passes);

  auto& instructions = predictor.runtime_program().instructions();
  auto& program_desc = predictor.program_desc();
  auto* scope = const_cast<lite::OpLite*>(instructions[0].op())->scope();

  CollectVarDescs(&(conf->var_descs), program_desc);
  PrepareModelInputTensor(*conf, scope, program_desc);
  predictor.Run();

  CollectAndDumpTopoInfo(instructions, *conf);
  CollectAndDumpTensorInfo(instructions, program_desc, *conf);

  // TODO(sangoly): Maybe add some profile info here
  auto* out = predictor.GetOutput(0);
  LOG(INFO) << out << " memory size " << out->data_size();
  LOG(INFO) << "out " << out->data<float>()[0];
  LOG(INFO) << "dims " << out->dims();
  LOG(INFO) << "out data size: " << out->data_size();
}

}  // namespace debug
}  // namespace tools
}  // namespace lite
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  paddle::lite::tools::debug::DebugConfig conf;
  paddle::lite::tools::debug::ParseConfig(&conf);
  paddle::lite::tools::debug::Run(&conf);

  return 0;
}
