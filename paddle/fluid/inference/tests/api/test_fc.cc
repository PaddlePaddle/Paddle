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
  inputs.resize(2);

  auto& input = inputs.front();
  input.shape.assign({FLAGS_batch_size, 129});
  input.data.Resize(FLAGS_batch_size * 129 * sizeof(float));
  auto* data = static_cast<float*>(input.data.data());
  for (int i = 0; i < 129* FLAGS_batch_size; i++) {
    data[i] = rand() / RAND_MAX;
  }

  auto& input2 = inputs.at(1);
  input2.shape.assign({FLAGS_batch_size, 81}); 
  input2.data.Resize(FLAGS_batch_size * 81 * sizeof(float));
  auto* data2 = static_cast<float*>(input2.data.data());
  for (int i = 0; i < 81 * FLAGS_batch_size; ++i) {
    data2[i] = rand() / RAND_MAX;
  }

  std::vector<PaddleTensor> outputs;

  inference::Timer timer;
  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    ASSERT_TRUE(predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "output: " << static_cast<float*>(outputs.front().data.data())[0];
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
  inputs.resize(2);

  auto& input = inputs.front();
  input.shape.assign({FLAGS_batch_size, 129});
  input.data.Resize(FLAGS_batch_size * 129 * sizeof(float));
  auto* data = static_cast<float*>(input.data.data());
  for (int i = 0; i < 129* FLAGS_batch_size; i++) {
    data[i] = rand() / RAND_MAX;
  }

  auto& input2 = inputs.at(1);
  input2.shape.assign({FLAGS_batch_size, 81}); 
  input2.data.Resize(FLAGS_batch_size * 81 * sizeof(float));
  auto* data2 = static_cast<float*>(input2.data.data());
  for (int i = 0; i < 81 * FLAGS_batch_size; ++i) {
    data2[i] = rand() / RAND_MAX;
  }

  std::vector<PaddleTensor> outputs;

  inference::Timer timer;
  timer.tic();
  for (int i = 0; i < FLAGS_repeat; i++) {
    ASSERT_TRUE(predictor->Run(inputs, &outputs));
  }
  LOG(INFO) << "Run old latency " << timer.toc() / FLAGS_repeat;

  LOG(INFO) << "output: " << static_cast<float*>(outputs.front().data.data())[0];
}


TEST(test, main) {
  FLAGS_global_use_lite_op = true;
  AnalysisConfig config;
  config.SetModel(FLAGS_model);
  config.SwitchIrOptim(true);
  config.SwitchIrDebug(false);

  auto predictor = CreatePaddlePredictor(config);

  std::vector<PaddleTensor> inputs;
  inputs.resize(2);

  auto& input = inputs.front();
  input.shape.assign({FLAGS_batch_size, 129});
  input.data.Resize(FLAGS_batch_size * 129 * sizeof(float));
  auto* data = static_cast<float*>(input.data.data());
  for (int i = 0; i < 129* FLAGS_batch_size; i++) {
    data[i] = rand() / RAND_MAX;
  }

  auto& input2 = inputs.at(1);
  input2.shape.assign({FLAGS_batch_size, 81}); 
  input2.data.Resize(FLAGS_batch_size * 81 * sizeof(float));
  auto* data2 = static_cast<float*>(input2.data.data());
  for (int i = 0; i < 81 * FLAGS_batch_size; ++i) {
    data2[i] = rand() / RAND_MAX;
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
  auto input_tensor = predictor->GetInputTensor("x_0");
  input_tensor->Reshape({FLAGS_batch_size, 129});
  auto* input_data = input_tensor->mutable_data<float>(PaddlePlace::kCPU);
  for (int i = 0; i < 129 * FLAGS_batch_size; i++) {
    input_data[i] = rand() / RAND_MAX;
  }

  auto input_tensor_1 = predictor->GetInputTensor("x_1");
  input_tensor_1->Reshape({FLAGS_batch_size, 81});
  float* input_data_1 = input_tensor_1->mutable_data<float>(PaddlePlace::kCPU);
  for (int i = 0; i < 81 * FLAGS_batch_size; i++) {
    input_data_1[i] = rand() / RAND_MAX;
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
