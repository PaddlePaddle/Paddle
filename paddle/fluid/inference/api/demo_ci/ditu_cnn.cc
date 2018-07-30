#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/paddle_inference_api.h"
#include "utils.h"

using namespace paddle;

DEFINE_string(modeldir, "", "Directory of the inference model.");

void PrepareInputs(std::vector<PaddleTensor>* inputs);

int Main(int max_batch) {
  NativeConfig config;
  config.model_dir = FLAGS_modeldir;
  config.use_gpu = false;
  config.device = 0;

  std::vector<std::vector<int>> shapes({{4},
                                        {1, 50, 12},
                                        {1, 50, 19},
                                        {1, 50, 1},
                                        {4, 50, 1},
                                        {1, 50, 1},
                                        {5, 50, 1},
                                        {7, 50, 1},
                                        {3, 50, 1}});

  std::vector<PaddleTensor> inputs;
  for (auto& shape : shapes) {
    PaddleTensor t_feature_f{
        .name = "xx",
        .shape = shape,
        .lod = std::vector<std::vector<size_t>>(),
        .data = PaddleBuf(max_batch * sizeof(float) *
                          std::accumulate(shapes[i].begin(), shape.end(), 1,
                                          [](int a, int b) { return a * b; })),
        .dtype = PaddleDType::FLOAT32};
  }

  auto predictor =
      CreatePaddlePredictor<NativeConfig, PaddleEngineKind::kNative>(config);

  // { batch begin

  std::vector<PaddleTensor> outputs;

  CHECK(predictor->Run(inputs, &outputs));

  // } batch end
}

int main() {
  Main(4);
  return 0;
}
