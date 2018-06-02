#include "paddle/contrib/inference/paddle_inference_api_anakin_engine.h"
#include <cuda.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {

PaddleInferenceAnakinPredictor::PaddleInferenceAnakinPredictor(
    const AnakinConfig &config) {
  PADDLE_ENFORCE(Init(config));
}

bool PaddleInferenceAnakinPredictor::Init(const AnakinConfig &config) {
  // TODO(Superjomn) Tell anakin to support return code.
  engine_.Build(config.model_file, config.max_batch_size);
  return true;
}

bool PaddleInferenceAnakinPredictor::Run(
    const std::vector<PaddleTensor> &inputs,
    std::vector<PaddleTensor> *output_data) {
  for (const auto &input : inputs) {
    PADDLE_ENFORCE(input.dtype == PaddleDType::FLOAT32);
    engine_.SetInputFromCPU(input.name, static_cast<float *>(input.data.data),
                            input.data.length);
  }

  // TODO(Superjomn) Tell anakin to support return code.
  engine_.Execute();

  PADDLE_ENFORCE(!output_data->empty(),
                 "At least one output should be set with tensors' names.");
  for (const auto &output : *output_data) {
    auto *tensor = engine_.GetOutputInGPU(output.name);
    output.shape = tensor->shape();
    // Copy data from GPU -> CPU
    PADDLE_ENFORCE_EQ(cudaMemcpy(output.data.data, tensor->data(),
                                 tensor->size(), cudaMemcpyDeviceToHost),
                      0);
  }
  return true;
}

// TODO(Superjomn) To implement latter.
std::unique_ptr<PaddlePredictor> PaddleInferenceAnakinPredictor::Clone() {
  return nullptr;
}

}  // namespace paddle