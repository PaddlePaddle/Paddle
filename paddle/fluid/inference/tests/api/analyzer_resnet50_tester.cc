// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/mkldnn_analyzer.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/api/timer.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(data_list, "", "Path to a file with a list of image files.");
DEFINE_string(data_dir, "", "Path to a directory with image files.");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(iterations, 1, "How many times to repeat run.");
// dimensions of imagenet images are assumed as default:
DEFINE_int32(height, 224, "Height of the image.");
DEFINE_int32(width, 224, "Width of the image.");
DEFINE_int32(channels, 3, "Width of the image.");

namespace paddle {

template <typename T>
void fill_data(std::unique_ptr<T[]>& data, unsigned int count)
{
  for (unsigned int i = 0; i< count; ++i) {
    *(data.get() + i) = i;
  }
}

void PrintResults(int bs, int iterations, double lat_avg, float acc_avg) {
  LOG(INFO) << "===========profile result===========";
  LOG(INFO) << "batch_size: " << bs << ", iterations: " << iterations
            << ", avg latency: " << lat_avg << "ms"
	    << ", avg accuracy: " << acc_avg;
  LOG(INFO) << "=====================================";
}

void Main(int batch_size) {
  
  auto count = [](std::vector<int>& shapevec)
  {
    auto sum = shapevec.size() > 0 ? 1 : 0;
    for (unsigned int i=0; i < shapevec.size(); ++i) {
      sum *= shapevec[i];
    }
    return sum;
  }; 

  // define input: data
  std::vector<int> shape;
  shape.push_back(FLAGS_batch_size);
  shape.push_back(FLAGS_channels);
  shape.push_back(FLAGS_height);
  shape.push_back(FLAGS_width);

  // use fake data
  std::unique_ptr<float[]> data(new float[count(shape)]);
  fill_data<float>(data, count(shape));

  paddle::PaddleTensor input;
  input.name = "xx";
  input.shape = shape,
  input.data = paddle::PaddleBuf(data.get(), count(shape)*sizeof(float)),
  input.dtype = paddle::PaddleDType::FLOAT32;

  std::cout << std::endl << "Executing model: " << FLAGS_infer_model << std::endl <<
  "Batch Size: " << FLAGS_batch_size << std::endl <<
  "Channels: " << FLAGS_channels << std::endl <<
  "Height: " << FLAGS_height << std::endl <<
  "Width: " << FLAGS_width << std::endl;

  // define input: labels
  int label_size = FLAGS_batch_size;
  std::unique_ptr<int64_t[]> label(new int64_t[label_size]);
  fill_data<int64_t>(label, label_size);

  paddle::PaddleTensor input_label;
  input_label.name = "yy",
  input_label.shape = std::vector<int>({label_size, 1}),
  input_label.data = paddle::PaddleBuf(label.get(), label_size*sizeof(int64_t)),
  input_label.dtype = paddle::PaddleDType::INT64;

  // create predictor
  MKLDNNAnalysisConfig config;
  config.model_dir = FLAGS_infer_model;
  config.use_gpu = false;
  config.enable_ir_optim = true;
  // add passes to execute keeping the order - without MKL-DNN
  config.ir_passes.push_back("fc_fuse_pass");
  // add passes to execute with MKL-DNN
  config.ir_mkldnn_passes.push_back("conv_relu_mkldnn_fuse_pass");
  config.ir_mkldnn_passes.push_back("fc_fuse_pass");
  auto predictor =
      CreatePaddlePredictor<MKLDNNAnalysisConfig, PaddleEngineKind::kAnalysis>(
          config);

  // define output
  std::vector<PaddleTensor> output_slots;

  // run prediction
  inference::Timer timer;
  double sum = 0;
  for (int i = 0; i < FLAGS_iterations; i++) {
    timer.tic();
    CHECK(predictor->Run({input, input_label}, &output_slots));
    sum += timer.toc();
  }

  // handle output
  CHECK_EQ(output_slots.size(), 2UL);
  PaddleTensor output = output_slots[0];
  CHECK_EQ(output.lod.size(), 0UL);
  CHECK_EQ(output.dtype, paddle::PaddleDType::FLOAT32);
  float *odata = static_cast<float*>(output.data.data());
  size_t olen = output.data.length() / sizeof(FLOAT32);
  float acc_avg = std::accumulate(odata, odata + olen, 0.0) / 10;
  double lat_avg = sum / FLAGS_iterations;

  PrintResults(batch_size, FLAGS_iterations, lat_avg, acc_avg);
}

TEST(resnet50, basic) { Main(FLAGS_batch_size); }

}  // namespace paddle

