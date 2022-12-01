// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tests/api/analyzer_ernie_tester.h"

namespace paddle {
namespace inference {

/*
 * this model is unreasonable, it set a middle-tensor persistable, so
 * ridiculous! so I disable constant_folding_pass
 */

using paddle::PaddleTensor;

#ifdef PADDLE_WITH_MKLDNN
void SetInt8Config(AnalysisConfig *cfg,
                   std::vector<paddle::PaddleTensor> data) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->EnableMKLDNN();
  cfg->DisableMkldnnFcPasses();  // fc passes caused loss in accuracy
  cfg->EnableMkldnnQuantizer();
  auto pass_builder = cfg->pass_builder();
  pass_builder->DeletePass("constant_folding_pass");
  auto warmup_data = std::make_shared<std::vector<PaddleTensor>>(data);
  cfg->mkldnn_quantizer_config()->SetWarmupData(warmup_data);
  cfg->mkldnn_quantizer_config()->SetWarmupBatchSize(FLAGS_batch_size);
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
}

// Compare result of NativeConfig and AnalysisConfig
void compare_int8(bool use_mkldnn = false) {
  std::vector<std::vector<PaddleTensor>> inputs;
  LoadInputData(&inputs);

  AnalysisConfig cfg;
  SetInt8Config(&cfg, inputs[0]);

  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), inputs);
}

TEST(Analyzer_ernie, compare_int8_mkldnn) {
  compare_int8(true /* use_mkldnn */);
}
#endif

}  // namespace inference
}  // namespace paddle
