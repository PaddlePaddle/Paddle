// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "test_helper.h"  // NOLINT
#include "test_suite.h"   // NOLINT

DEFINE_string(modeldir, "", "Directory of the inference model.");

namespace paddle_infer {

paddle::test::Record PrepareInput(int batch_size, int shape_size = 224) {
  // init input data
  int channel = 3;
  int width = shape_size;   // w = 224
  int height = shape_size;  // h = 224
  paddle::test::Record image_Record;
  int input_num = batch_size * channel * width * height;
  std::vector<float> input_data(input_num, 1);
  image_Record.data = input_data;
  image_Record.shape = std::vector<int>{batch_size, channel, width, height};
  image_Record.type = paddle::PaddleDType::FLOAT32;
  return image_Record;
}

TEST(tensorrt_tester_mobilenetv1, tuned_dynamic_trt_fp32_bz2) {
  bool tuned_shape = true;
  std::string shape_range_info = FLAGS_modeldir + "/shape_range_info.pbtxt";
  LOG(INFO) << "tensorrt tuned info saved to " << shape_range_info;

  // init input data
  std::map<std::string, paddle::test::Record> my_input_data_map;
  my_input_data_map["x"] = PrepareInput(2, 448);
  // init output data
  std::map<std::string, paddle::test::Record> infer_output_data,
      truth_output_data;
  if (tuned_shape) {
    // NOTE: shape_range_info will be saved after destructor of predictor
    // function
    // prepare ground truth config
    paddle_infer::Config tune_config;
    tune_config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                         FLAGS_modeldir + "/inference.pdiparams");
    tune_config.SwitchIrOptim(false);
    tune_config.EnableUseGpu(1000, 0);
    tune_config.CollectShapeRangeInfo(shape_range_info);

    auto predictor_tune = paddle_infer::CreatePredictor(tune_config);
    SingleThreadPrediction(
        predictor_tune.get(), &my_input_data_map, &truth_output_data, 1);
  }

  // prepare inference config
  paddle_infer::Config config;
  config.SetModel(FLAGS_modeldir + "/inference.pdmodel",
                  FLAGS_modeldir + "/inference.pdiparams");
  config.EnableUseGpu(1000, 0);
  config.EnableTensorRtEngine(
      1 << 20, 2, 5, paddle_infer::PrecisionType::kFloat32, false, false);
  config.EnableTunedTensorRtDynamicShape(shape_range_info, true);
  LOG(INFO) << config.Summary();
  paddle_infer::services::PredictorPool pred_pool(config, 1);
  SingleThreadPrediction(
      pred_pool.Retrieve(0), &my_input_data_map, &infer_output_data);
  // check outputs
  CompareRecord(&truth_output_data, &infer_output_data);
  VLOG(1) << "finish test";
}

}  // namespace paddle_infer

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
