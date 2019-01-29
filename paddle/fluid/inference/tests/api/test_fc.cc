#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "time.h"

using namespace paddle;

DEFINE_string(model, "/home/chunwei/project2/models/fc/fluid_checkpoint", "");

TEST(test, main) {
  contrib::AnalysisConfig config;
  config.SetModel(FLAGS_model);
  config.SwitchIrOptim(true);
  config.SwitchIrDebug(true);
  //config.EnableMKLDNN();
  config.pass_builder()->TurnOnDebug();

  auto predictor = CreatePaddlePredictor(config);

  std::vector<PaddleTensor> inputs;
  inputs.resize(1);

  auto& input = inputs.front();
  input.shape.assign({FLAGS_batch_size, 210});
  input.data.Resize(FLAGS_batch_size * 210 * sizeof(float));
  auto* data = static_cast<float*>(input.data.data());
  for (int i = 0; i < 210*FLAGS_batch_size; i++) {
    data[i] = rand() / RAND_MAX;
  }

  std::vector<PaddleTensor> outputs;

  inference::Timer timer;
  timer.tic();
  for (int i = 0; i < 1000; i++) {
    ASSERT_TRUE(predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "latency " << timer.toc() / 1000;
}
