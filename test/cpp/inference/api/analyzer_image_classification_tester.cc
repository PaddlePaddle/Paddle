/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "test/cpp/inference/api/tester_helper.h"

PD_DEFINE_bool(disable_mkldnn_fc, false, "Disable usage of MKL-DNN's FC op");

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model", FLAGS_infer_model + "/params");
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
  cfg->DeletePass("constant_folding_pass");
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  SetFakeImageInput(inputs, FLAGS_infer_model);
}

// Easy for profiling independently.
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  if (use_mkldnn) {
    cfg.EnableMKLDNN();
    if (FLAGS_disable_mkldnn_fc) {
      cfg.DisableMkldnnFcPasses();
    }
  }
  std::vector<std::vector<PaddleTensor>> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all,
                 &outputs,
                 FLAGS_num_threads);
}

TEST(Analyzer_resnet50, profile) { profile(); }
#ifdef PADDLE_WITH_DNNL
TEST(Analyzer_resnet50, profile_mkldnn) { profile(true /* use_mkldnn */); }
#endif

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
    if (FLAGS_disable_mkldnn_fc) {
      cfg.DisableMkldnnFcPasses();
    }
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(Analyzer_resnet50, compare) { compare(); }
#ifdef PADDLE_WITH_DNNL
TEST(Analyzer_resnet50, compare_mkldnn) { compare(true /* use_mkldnn */); }
#endif

// Compare Deterministic result
TEST(Analyzer_resnet50, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
