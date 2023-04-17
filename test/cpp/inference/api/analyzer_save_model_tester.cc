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

#include "paddle/fluid/inference/analysis/helper.h"
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

/*
 * this model is unreasonable, it set a output tensor persistable, so
 * ridiculous! so I disable constant_folding_pass
 */

TEST(Analyzer, save_model) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  cfg.SetModel(FLAGS_infer_model + "/__model__", FLAGS_infer_model + "/param");

  auto pass_builder = cfg.pass_builder();
  pass_builder->DeletePass("constant_folding_pass");

  //  ensure the path being unique
  std::string optimModelPath = FLAGS_infer_model + "/only_for_save_model_test";
  MKDIR(optimModelPath.c_str());
  SaveOptimModel(&cfg, optimModelPath);

  // Each config can only be applied to one predictor.
  AnalysisConfig cfg2;
  SetConfig(&cfg2);
  cfg2.pass_builder()->ClearPasses();
  cfg2.SetModel(optimModelPath + "/model", optimModelPath + "/params");
  int origin_num_ops = GetNumOps(cfg2);

  AnalysisConfig cfg3;
  SetConfig(&cfg3);
  auto pass_builder3 = cfg3.pass_builder();
  pass_builder3->DeletePass("constant_folding_pass");
  cfg3.SetModel(optimModelPath + "/model", optimModelPath + "/params");
  int fused_num_ops = GetNumOps(cfg3);
  CHECK_LE(fused_num_ops, origin_num_ops);
}

}  // namespace inference
}  // namespace paddle
