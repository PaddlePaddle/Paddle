/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "test/cpp/inference/api/trt_test_helper.h"

namespace paddle_infer {
int GetTrtVersion() {
  auto trt_runtime_version = GetTrtRuntimeVersion();
  return std::get<0>(trt_runtime_version) * 1000 +
         std::get<1>(trt_runtime_version) * 100 +
         std::get<2>(trt_runtime_version);
  return 0;
}
}  // namespace paddle_infer
namespace paddle {
std::vector<std::string> GetClosedPasses(
    const std::vector<std::string> &origin_passes,
    const std::vector<std::string> &ctrl_passes) {
  std::vector<std::string> closed_passes;
  for (const auto &pass_name : origin_passes) {
    auto it = std::find(ctrl_passes.begin(), ctrl_passes.end(), pass_name);
    if (it == ctrl_passes.end()) {
      closed_passes.push_back(pass_name);
    }
  }
  return closed_passes;
}

TEST(GPU, gpu_pass_ctrl_test_fp32) {
  AnalysisConfig config;
  config.EnableUseGpu(100, 0, AnalysisConfig::Precision::kHalf);
  config.EnablePassController(true);
  auto pass_builder = config.pass_builder();
  auto orgin_passes = pass_builder->AllPasses();
  auto pass_controller = config.pass_controller();
  auto ctrl_passes =
      pass_controller->GetCtrlPassList(pass_builder->AllPasses(),
                                       static_cast<int64_t>(0),
                                       static_cast<int64_t>(0),
                                       config.use_gpu(),
                                       config.tensorrt_engine_enabled());
  auto closed_passes = GetClosedPasses(orgin_passes, ctrl_passes);
  ASSERT_EQ(closed_passes.size(), size_t(0));
}

TEST(TRT, trt_pass_ctrl_test_fp16) {
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.EnablePassController(true);
  config.EnableTensorRtEngine(
      1 << 30, 1, 5, AnalysisConfig::Precision::kHalf, false, false);
  auto pass_builder = config.pass_builder();
  auto orgin_passes = pass_builder->AllPasses();
  auto pass_controller = config.pass_controller();
  auto ctrl_passes =
      pass_controller->GetCtrlPassList(pass_builder->AllPasses(),
                                       static_cast<int64_t>(0),
                                       static_cast<int64_t>(2),
                                       config.use_gpu(),
                                       config.tensorrt_engine_enabled());
  auto closed_passes = GetClosedPasses(orgin_passes, ctrl_passes);
  auto trt_runtime_version = paddle_infer::GetTrtVersion();
  if (trt_runtime_version >= 8600) {
    ASSERT_EQ(closed_passes.size(), size_t(10));
    auto it = std::find(closed_passes.begin(),
                        closed_passes.end(),
                        "preln_residual_bias_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "trt_skip_layernorm_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(
        closed_passes.begin(), closed_passes.end(), "vit_attention_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "layernorm_shift_partition_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(
        closed_passes.begin(), closed_passes.end(), "reverse_roll_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "preln_layernorm_x_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "split_layernorm_to_math_ops_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(
        closed_passes.begin(), closed_passes.end(), "add_support_int8_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "merge_layernorm_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "elementwiseadd_transpose_pass");
    ASSERT_EQ(it, closed_passes.end());
  } else {
    ASSERT_EQ(closed_passes.size(), size_t(1));
    ASSERT_EQ(closed_passes[0], "add_support_int8_pass");
  }
}

TEST(TRT, trt_pass_ctrl_test_fp32) {
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.EnablePassController(true);
  config.EnableTensorRtEngine(
      1 << 30, 1, 5, AnalysisConfig::Precision::kFloat32, false, false);
  auto pass_builder = config.pass_builder();
  auto orgin_passes = pass_builder->AllPasses();
  auto pass_controller = config.pass_controller();
  auto ctrl_passes =
      pass_controller->GetCtrlPassList(pass_builder->AllPasses(),
                                       static_cast<int64_t>(0),
                                       static_cast<int64_t>(0),
                                       config.use_gpu(),
                                       config.tensorrt_engine_enabled());
  auto trt_runtime_version = paddle_infer::GetTrtVersion();
  auto closed_passes = GetClosedPasses(orgin_passes, ctrl_passes);
  if (trt_runtime_version >= 8600) {
    ASSERT_EQ(closed_passes.size(), size_t(9));
    auto it = std::find(closed_passes.begin(),
                        closed_passes.end(),
                        "preln_residual_bias_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(
        closed_passes.begin(), closed_passes.end(), "vit_attention_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "layernorm_shift_partition_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(
        closed_passes.begin(), closed_passes.end(), "reverse_roll_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "preln_layernorm_x_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "split_layernorm_to_math_ops_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(
        closed_passes.begin(), closed_passes.end(), "add_support_int8_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "merge_layernorm_fuse_pass");
    ASSERT_EQ(it, closed_passes.end());
    it = std::find(closed_passes.begin(),
                   closed_passes.end(),
                   "elementwiseadd_transpose_pass");
    ASSERT_EQ(it, closed_passes.end());
  } else {
    ASSERT_EQ(closed_passes.size(), size_t(1));
    ASSERT_EQ(closed_passes[0], "add_support_int8_pass");
  }
}

}  // namespace paddle
