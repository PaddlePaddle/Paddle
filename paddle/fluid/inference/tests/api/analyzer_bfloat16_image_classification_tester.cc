/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>

#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "paddle/fluid/platform/cpu_info.h"

DEFINE_bool(enable_mkldnn, true, "Enable MKLDNN");

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  std::ifstream model_file(FLAGS_infer_model + "/__model__");
  if (model_file.good())
    cfg->SetModel(FLAGS_infer_model);
  else
    cfg->SetModel(FLAGS_infer_model + "/inference.pdmodel",
                  FLAGS_infer_model + "/inference.pdiparams");
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_num_threads);
  if (FLAGS_enable_mkldnn) cfg->EnableMKLDNN();
}

TEST(Analyzer_bfloat16_image_classification, bfloat16) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  AnalysisConfig b_cfg;
  SetConfig(&b_cfg);

  // read data from file and prepare batches with test data
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInputs(&input_slots_all);
  if (FLAGS_enable_mkldnn && FLAGS_enable_bf16 &&
      platform::MayIUse(platform::cpu_isa_t::avx512_bf16)) {
    b_cfg.EnableMkldnnBfloat16();
  } else {
    FLAGS_enable_bf16 = false;
  }
  CompareBFloat16AndAnalysis(&cfg, &b_cfg, input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
