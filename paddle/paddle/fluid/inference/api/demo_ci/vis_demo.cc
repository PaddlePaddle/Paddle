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

/*
 * This file contains demo for mobilenet, se-resnext50 and ocr.
 */

#include <glog/logging.h>

#include "gflags/gflags.h"
#include "utils.h"  // NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_string(refer, "", "path to reference result for comparison.");
DEFINE_string(data,
              "",
              "path of data; each line is a record, format is "
              "'<space split floats as data>\t<space split ints as shape'");
DEFINE_bool(use_gpu, false, "Whether use gpu.");

namespace paddle {
namespace demo {

/*
 * Use the native and analysis fluid engine to inference the demo.
 */
void Main(bool use_gpu) {
  std::unique_ptr<PaddlePredictor> predictor, analysis_predictor;
  AnalysisConfig config;
  if (use_gpu) {
    config.EnableUseGpu(100, 0);
  }
  config.SetModel(FLAGS_modeldir + "/__model__",
                  FLAGS_modeldir + "/__params__");

  predictor = CreatePaddlePredictor<NativeConfig>(config.ToNativeConfig());
  analysis_predictor = CreatePaddlePredictor(config);

  // Just a single batch of data.
  std::string line;
  std::ifstream file(FLAGS_data);
  std::getline(file, line);
  auto record = ProcessALine(line);
  file.close();

  // Inference.
  PaddleTensor input;
  input.shape = record.shape;
  input.data =
      PaddleBuf(record.data.data(), record.data.size() * sizeof(float));
  input.dtype = PaddleDType::FLOAT32;

  std::vector<PaddleTensor> output, analysis_output;
  predictor->Run({input}, &output, 1);

  auto& tensor = output.front();

  // compare with reference result
  CheckOutput(FLAGS_refer, tensor, 1e-4);

  // the analysis_output has some diff with native_output,
  // TODO(luotao): add CheckOutput for analysis_output later.
  analysis_predictor->Run({input}, &analysis_output, 1);
}

}  // namespace demo
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_use_gpu) {
    paddle::demo::Main(true /*use_gpu*/);
  } else {
    paddle::demo::Main(false /*use_gpu*/);
  }
  return 0;
}
