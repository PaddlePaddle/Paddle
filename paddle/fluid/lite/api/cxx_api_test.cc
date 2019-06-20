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
#include "paddle/fluid/lite/api/lite_api_test_helper.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/mir/use_passes.h"
#include "paddle/fluid/lite/core/op_registry.h"
#include "paddle/fluid/lite/kernels/use_kernels.h"
#include "paddle/fluid/lite/operators/use_ops.h"

// For training.
DEFINE_string(startup_program_path, "", "");
DEFINE_string(main_program_path, "", "");

// for eval
DEFINE_string(eval_model_dir, "", "");

namespace paddle {
namespace lite {

#ifndef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
TEST(CXXApi, test) {
  const lite::Tensor* out = RunHvyModel();
  LOG(INFO) << out << " memory size " << out->data_size();
  for (int i = 0; i < 10; i++) {
    LOG(INFO) << "out " << out->data<float>()[i];
  }
  LOG(INFO) << "dims " << out->dims();
  // LOG(INFO) << "out " << *out;
}

TEST(CXXApi, save_model) {
  lite::ExecutorLite predictor;
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                   Place{TARGET(kX86), PRECISION(kFloat)}});
  predictor.Build(FLAGS_model_dir, Place{TARGET(kCUDA), PRECISION(kFloat)},
                  valid_places);

  LOG(INFO) << "Save optimized model to " << FLAGS_optimized_model;
  predictor.SaveModel(FLAGS_optimized_model);
}

/*TEST(CXXTrainer, train) {
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

  // LOG(INFO) << main_program_desc.DebugString();

  for (const auto& op : main_program_desc.blocks(0).ops()) {
    LOG(INFO) << "get op " << op.type();
  }

  return;

  trainer.RunStartupProgram(startup_program_desc);
  auto& exe = trainer.BuildMainProgramExecutor(main_program_desc);
  auto* tensor0 = exe.GetInput(0);
  tensor0->Resize(std::vector<int64_t>({100, 100}));
  auto* data0 = tensor0->mutable_data<float>();
  data0[0] = 0;

  exe.Run();
}*/
#endif  // LITE_WITH_LIGHT_WEIGHT_FRAMEWORK

#ifdef LITE_WITH_ARM
TEST(CXXApi, eval) {
  DeviceInfo::Init();
  lite::ExecutorLite predictor;
  std::vector<Place> valid_places({Place{TARGET(kHost), PRECISION(kFloat)},
                                   Place{TARGET(kARM), PRECISION(kFloat)}});

  predictor.Build(FLAGS_eval_model_dir, Place{TARGET(kARM), PRECISION(kFloat)},
                  valid_places);

  auto* input_tensor = predictor.GetInput(0);
  input_tensor->Resize(DDim(std::vector<DDim::value_type>({1, 3, 224, 224})));
  auto* data = input_tensor->mutable_data<float>();
  for (int i = 0; i < input_tensor->dims().production(); i++) {
    data[i] = 1;
  }

  predictor.Run();

  auto* out = predictor.GetOutput(0);
  std::vector<float> results({0.00097802, 0.00099822, 0.00103093, 0.00100121,
                              0.00098268, 0.00104065, 0.00099962, 0.00095181,
                              0.00099694, 0.00099406});
  for (int i = 0; i < results.size(); ++i) {
    EXPECT_NEAR(out->data<float>()[i], results[i], 1e-5);
  }
  ASSERT_EQ(out->dims().size(), 2);
  ASSERT_EQ(out->dims()[0], 1);
  ASSERT_EQ(out->dims()[1], 1000);
}
#endif

}  // namespace lite
}  // namespace paddle
