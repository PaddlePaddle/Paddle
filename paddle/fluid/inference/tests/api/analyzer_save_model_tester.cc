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

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SwitchSpecifyInputNames();
  cfg->SwitchIrOptim(true);
  cfg->SwitchIrDebug();
}

int GetNumOps(const AnalysisConfig &cfg) {
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  GetFuseStatis(static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  return num_ops;
}

TEST(Analyzer, save_model) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  cfg.SetModel(FLAGS_infer_model + "/__model__", FLAGS_infer_model + "/param");
  std::string optimModelPath = FLAGS_infer_model + "/saved_optim_model";
  mkdir(optimModelPath.c_str(), 0777);
  SaveOptimModel(&cfg, optimModelPath);

  cfg.pass_builder()->ClearPasses();
  int origin_num_ops = GetNumOps(cfg);
  cfg.SetModel(optimModelPath + "/model", optimModelPath + "/params");
  int fused_num_ops = GetNumOps(cfg);
  CHECK_LE(fused_num_ops, origin_num_ops);
}

}  // namespace inference
}  // namespace paddle
