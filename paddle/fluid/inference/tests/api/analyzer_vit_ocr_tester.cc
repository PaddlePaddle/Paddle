/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg, bool use_mkldnn = false) {
  cfg->SetModel(FLAGS_infer_model + "/inference.pdmodel",
                FLAGS_infer_model + "/inference.pdiparams");

  cfg->DisableGpu();
  if (use_mkldnn) {
    cfg->EnableMKLDNN();
    cfg->SwitchIrOptim();

    size_t insertingIndex = cfg->pass_builder()->GetPassIndex(
        "fc_elementwise_add_mkldnn_fuse_pass");
    cfg->pass_builder()->InsertPass(insertingIndex, "fc_act_mkldnn_fuse_pass");
    cfg->pass_builder()->InsertPass(insertingIndex, "fc_mkldnn_pass");
  }
}

// Check the fuse status
TEST(Analyzer_vit_ocr, fuse_status) {
  AnalysisConfig cfg;
  SetConfig(&cfg, true);
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_status = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);

  CHECK_EQ(fuse_status.at("fc_mkldnn_pass"), 33);
  CHECK_EQ(fuse_status.at("conv_activation_mkldnn_fuse"), 2);
  CHECK_EQ(fuse_status.at("fc_elementwise_add_mkldnn_fuse"), 16);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
