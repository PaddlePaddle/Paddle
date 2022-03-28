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

/*
 * This file contains demo of mobilenet for tensorrt.
 */

#include <glog/logging.h>  // use glog instead of CHECK to avoid importing other paddle header files.
#include <vector>
#include "gflags/gflags.h"
#include "utils.h"  // NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");

namespace paddle {
namespace demo {

/*
 * Use the onnxruntime engine to inference the demo.
 */
void Main() {
  paddle::AnalysisConfig config;
  config.EnableONNXRuntime();
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  auto predictor = paddle_infer::CreatePredictor(config);

  // Inference.
  std::vector<int> input_shape = {1, 3, 224, 224};
  std::vector<float> input_data(1 * 3 * 224 * 224, 1.0);
  std::vector<float> out_data;
  out_data.resize(1000);
  auto input_names = predictor->GetInputNames();
  auto output_names = predictor->GetOutputNames();
  auto input_tensor = predictor->GetInputHandle(input_names[0]);
  input_tensor->Reshape(input_shape);
  auto output_tensor = predictor->GetOutputHandle(output_names[0]);

  input_tensor->CopyFromCpu(input_data.data());
  predictor->Run();
  output_tensor->CopyToCpu(out_data.data());

  VLOG(3) << "output.size " << out_data.size();
}

}  // namespace demo
}  // namespace paddle

int main(int argc, char** argv) {
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  paddle::demo::Main();
  return 0;
}
