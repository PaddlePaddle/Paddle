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

#include "paddle/fluid/lite/api/cxx_api.h"
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/lite/core/mir/passes.h"
#include "paddle/fluid/lite/core/op_registry.h"

DEFINE_string(model_dir, "", "");
DEFINE_string(optimized_model, "", "");

// For training.
DEFINE_string(startup_program_path, "", "");
DEFINE_string(main_program_path, "", "");

namespace paddle {
namespace lite {

TEST(CXXApi, test) {
  lite::ExecutorLite predictor;
#ifndef LITE_WITH_CUDA
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)}});
#else
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kCUDA), PRECISION(kFloat), DATALAYOUT(kNCHW)},
      Place{TARGET(kCUDA), PRECISION(kAny), DATALAYOUT(kNCHW)},
      Place{TARGET(kHost), PRECISION(kAny), DATALAYOUT(kNCHW)},
      Place{TARGET(kCUDA), PRECISION(kAny), DATALAYOUT(kAny)},
      Place{TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)},
  });
#endif

  predictor.Build(FLAGS_model_dir, Place{TARGET(kCUDA), PRECISION(kFloat)},
                  valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({100, 100})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < 100 * 100; i++) {
    data[i] = i;
  }

  // LOG(INFO) << "input " << *input_tensor;

  predictor.Run();

  auto* out = predictor.GetOutput(0);
  LOG(INFO) << out << " memory size " << out->data_size();
  LOG(INFO) << "out " << out->data<float>()[0];
  LOG(INFO) << "out " << out->data<float>()[1];
  LOG(INFO) << "dims " << out->dims();
  // LOG(INFO) << "out " << *out;
}

#ifndef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
TEST(CXXApi, save_model) {
  lite::ExecutorLite predictor;
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)}});
  predictor.Build(FLAGS_model_dir, Place{TARGET(kCUDA), PRECISION(kFloat)},
                  valid_places);

  predictor.SaveModel(FLAGS_optimized_model);
}
#endif  // LITE_WITH_LIGHT_WEIGHT_FRAMEWORK

#ifndef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
TEST(CXXTrainer, train) {
  Place prefer_place({TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW)});
  std::vector<Place> valid_places({prefer_place});
  auto scope = std::make_shared<lite::Scope>();

  CXXTrainer trainer(scope, prefer_place, valid_places);

  std::string main_program_pb, startup_program_pb;
  ReadBinaryFile(FLAGS_main_program_path, &main_program_pb);
  ReadBinaryFile(FLAGS_startup_program_path, &startup_program_pb);
  framework::proto::ProgramDesc main_program_desc, startup_program_desc;
  main_program_desc.ParseFromString(main_program_pb);
  startup_program_desc.ParseFromString(startup_program_pb);

  LOG(INFO) << "startup: " << startup_program_desc.DebugString();
  LOG(INFO) << "main: " << main_program_desc.DebugString();

  LOG(INFO) << startup_program_desc.blocks_size();
  LOG(INFO) << main_program_desc.blocks_size();

  trainer.RunStartupProgram(startup_program_desc);
  auto& exe = trainer.BuildMainProgramExecutor(main_program_desc);
  auto* tensor0 = exe.GetInput(0);
  tensor0->Resize(std::vector<int64_t>({100, 100}));
  auto* data0 = tensor0->mutable_data<float>();
  data0[0] = 0;

  exe.Run();
}
#endif  // LITE_WITH_LIGHT_WEIGHT_FRAMEWORK

}  // namespace lite
}  // namespace paddle

USE_LITE_OP(mul);
USE_LITE_OP(fc);
USE_LITE_OP(scale);
USE_LITE_OP(feed);
USE_LITE_OP(fetch);
USE_LITE_OP(io_copy);
USE_LITE_KERNEL(fc, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kHost, kFloat, kNCHW, def);
USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);

#ifdef LITE_WITH_CUDA
USE_LITE_KERNEL(mul, kCUDA, kFloat, kNCHW, def);
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, device_to_host);
#endif
