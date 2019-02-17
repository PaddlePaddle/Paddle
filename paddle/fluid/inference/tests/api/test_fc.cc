#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "time.h"

using namespace paddle;

DEFINE_string(model, "/home/chunwei/project2/models/fc/fluid_checkpoint", "");

TEST(test, naive) {
  FLAGS_global_use_lite_op = false;
  AnalysisConfig config;
  config.SetModel(FLAGS_model);
  config.SwitchIrOptim(false);
  config.SwitchIrDebug(false);

  auto predictor = CreatePaddlePredictor(config);

  std::vector<PaddleTensor> inputs;
  inputs.resize(1);

  auto& input = inputs.front();
  input.shape.assign({FLAGS_batch_size, 210});
  input.data.Resize(FLAGS_batch_size * 210 * sizeof(float));
  auto* data = static_cast<float*>(input.data.data());
  for (int i = 0; i < 210; i++) {
    data[i] = rand() / RAND_MAX;
  }

  std::vector<PaddleTensor> outputs;

  inference::Timer timer;
  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    ASSERT_TRUE(predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "Naive Run latency " << timer.toc() / FLAGS_repeat;
}

TEST(test, main_old) {
  FLAGS_global_use_lite_op = false;
  AnalysisConfig config;
  config.SetModel(FLAGS_model);
  config.SwitchIrOptim(true);
  config.SwitchIrDebug(false);

  auto predictor = CreatePaddlePredictor(config);

  std::vector<PaddleTensor> inputs;
  inputs.resize(1);

  auto& input = inputs.front();
  input.shape.assign({FLAGS_batch_size, 210});
  input.data.Resize(FLAGS_batch_size * 210 * sizeof(float));
  auto* data = static_cast<float*>(input.data.data());
  for (int i = 0; i < 210; i++) {
    data[i] = rand() / RAND_MAX;
  }

  std::vector<PaddleTensor> outputs;

  inference::Timer timer;
  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    ASSERT_TRUE(predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "Run old latency " << timer.toc() / FLAGS_repeat;
}


TEST(test, main) {
  FLAGS_global_use_lite_op = true;
  AnalysisConfig config;
  config.SetModel(FLAGS_model);
  config.SwitchIrOptim(true);
  config.SwitchIrDebug(false);

  auto predictor = CreatePaddlePredictor(config);

  std::vector<PaddleTensor> inputs;
  inputs.resize(1);

  auto& input = inputs.front();
  input.shape.assign({FLAGS_batch_size, 210});
  input.data.Resize(FLAGS_batch_size * 210 * sizeof(float));
  auto* data = static_cast<float*>(input.data.data());
  for (int i = 0; i < 210; i++) {
    data[i] = rand() / RAND_MAX;
  }

  std::vector<PaddleTensor> outputs;

  inference::Timer timer;
  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    ASSERT_TRUE(predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "Run latency " << timer.toc() / FLAGS_repeat;
}

TEST(test, zero) {
  FLAGS_global_use_lite_op = true;
  AnalysisConfig config;
  config.SetModel(FLAGS_model);
  config.SwitchIrOptim(true);
  config.SwitchIrDebug(false);
  config.SwitchUseFeedFetchOps(false);

  auto predictor = CreatePaddlePredictor(config);

  LOG(INFO) << "batch_size " << FLAGS_batch_size;
  // prepare input data
  auto input_tensor = predictor->GetInputTensor("x");
  input_tensor->Reshape({FLAGS_batch_size, 210});
  auto* input_data = input_tensor->mutable_data<float>(PaddlePlace::kCPU);
  for (int i = 0; i < 210; i++) {
    input_data[i] = rand() / RAND_MAX;
  }

  // zerocopy run

  inference::Timer timer;
  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    predictor->ZeroCopyRun();
  }

  LOG(INFO) << "zero-copy run " << timer.toc() / FLAGS_repeat;

  // get output
  // ...
}
