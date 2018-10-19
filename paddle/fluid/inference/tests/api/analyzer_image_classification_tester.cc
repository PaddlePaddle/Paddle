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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of PADDLE_ENFORCE to avoid importing other paddle header files.
#include <gtest/gtest.h>
#include <random>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/inference/tests/api/data_reader.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(data_list, "", "Path to a file with a list of image files.");
DEFINE_string(data_dir, "", "Path to a directory with image files.");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(iterations, 1, "How many times to repeat run.");
DEFINE_int32(skip_batch_num, 0, "How many minibatches to skip in statistics.");
// dimensions of imagenet images are assumed as default:
DEFINE_int32(height, 224, "Height of the image.");
DEFINE_int32(width, 224, "Width of the image.");
DEFINE_int32(channels, 3, "Width of the image.");
DEFINE_bool(use_fake_data, false, "Use fake data (1,2,...).");
DEFINE_bool(skip_passes, false, "Skip running passes.");
DEFINE_bool(debug_display_images, false, "Show images in windows for debug.");
DECLARE_bool(profile);

namespace paddle {

template <typename T>
void fill_data(T* data, unsigned int count) {
  for (unsigned int i = 0; i < count; ++i) {
    *(data + i) = i;
  }
}

template <>
void fill_data<float>(float* data, unsigned int count) {
  static unsigned int seed = std::random_device()();
  static std::minstd_rand engine(seed);
  float mean = 0;
  float std = 1;
  std::normal_distribution<float> dist(mean, std);
  for (unsigned int i = 0; i < count; ++i) {
    data[i] = dist(engine);
  }
}

template <typename T>
void SkipFirstNData(std::vector<T>& v, int n) {
  std::vector<T>(v.begin() + FLAGS_skip_batch_num, v.end()).swap(v);
}

template <typename T>
T FindAverage(const std::vector<T>& v) {
  CHECK_GE(v.size(), 0);
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template <typename T>
T FindPercentile(std::vector<T> v, int p) {
  CHECK_GE(v.size(), 0);
  std::sort(v.begin(), v.end());
  if (p == 100) return v.back();
  int i = v.size() * p / 100;
  return v[i];
}

template <typename T>
T FindStandardDev(std::vector<T> v) {
  CHECK_GE(v.size(), 0);
  T mean = FindAverage(v);
  T var = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    var += (v[i] - mean) * (v[i] - mean);
  }
  var /= v.size();
  T std = sqrt(var);
  return std;
}

void PostprocessBenchmarkData(std::vector<double> latencies,
                              std::vector<double> fpses, double total_time_sec,
                              int total_samples) {
  // get rid of the first FLAGS_skip_batch_num data
  SkipFirstNData(latencies, FLAGS_skip_batch_num);
  SkipFirstNData(fpses, FLAGS_skip_batch_num);

  double lat_avg = FindAverage(latencies);
  double fps_avg = FindAverage(fpses);

  double lat_pc99 = FindPercentile(latencies, 99);
  double fps_pc01 = FindPercentile(fpses, 1);

  double lat_std = FindStandardDev(latencies);
  double fps_std = FindStandardDev(fpses);

  float examples_per_sec = total_samples / total_time_sec;

  printf("\n\nAvg fps: %.5f, std fps: %.5f, fps for 99pc latency: %.5f\n",
         fps_avg, fps_std, fps_pc01);
  printf("Avg latency: %.5f, std latency: %.5f, 99pc latency: %.5f\n", lat_avg,
         lat_std, lat_pc99);
  printf("Total examples: %d, total time: %.5f, total examples/sec: %.5f\n",
         total_samples, total_time_sec, examples_per_sec);
}

void Main() {
  auto count = [](std::vector<int>& shapevec) {
    auto sum = shapevec.size() > 0 ? 1 : 0;
    for (unsigned int i = 0; i < shapevec.size(); ++i) {
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
  paddle::PaddleTensor input_data;
  input_data.name = "xx";
  input_data.shape = shape;

  // define input: label
  int label_size = FLAGS_batch_size;
  paddle::PaddleTensor input_label;
  input_label.data.Resize(label_size * sizeof(int64_t));
  input_label.name = "yy";
  input_label.shape = std::vector<int>({label_size, 1});
  input_label.dtype = paddle::PaddleDType::INT64;

  CHECK_GE(FLAGS_iterations, 0);
  CHECK_GE(FLAGS_skip_batch_num, 0);

  // reader instance for not fake data
  std::unique_ptr<DataReader> reader;
  bool convert_to_rgb = true;

  // Read first batch
  if (FLAGS_use_fake_data) {
    // create fake data
    input_data.data.Resize(count(shape) * sizeof(float));
    fill_data<float>(static_cast<float*>(input_data.data.data()), count(shape));

    input_data.dtype = paddle::PaddleDType::FLOAT32;

    std::cout << std::endl
              << "Executing model: " << FLAGS_infer_model << std::endl
              << "Batch Size: " << FLAGS_batch_size << std::endl
              << "Channels: " << FLAGS_channels << std::endl
              << "Height: " << FLAGS_height << std::endl
              << "Width: " << FLAGS_width << std::endl;

    // create fake label
    fill_data<int64_t>(static_cast<int64_t*>(input_label.data.data()),
                       label_size);
  } else {
    reader.reset(new DataReader(FLAGS_data_list, FLAGS_data_dir, FLAGS_width,
                                FLAGS_height, FLAGS_channels, convert_to_rgb));
    if (!reader->SetSeparator('\t')) reader->SetSeparator(' ');
    // get imagenet data and label
    input_data.data.Resize(count(shape) * sizeof(float));
    input_data.dtype = PaddleDType::FLOAT32;

    reader->NextBatch(static_cast<float*>(input_data.data.data()),
                      static_cast<int64_t*>(input_label.data.data()),
                      FLAGS_batch_size, FLAGS_debug_display_images);
  }

  // create predictor
  contrib::AnalysisConfig config;
  // MKLDNNAnalysisConfig config;
  config.param_file = FLAGS_infer_model + "/params";
  config.prog_file = FLAGS_infer_model + "/model";
  // include mode: define which passes to include
  config.SetIncludeMode();
  config.use_gpu = false;
  config.enable_ir_optim = true;
  if (!FLAGS_skip_passes) {
    // add passes to execute keeping the order - without MKL-DNN
    config.ir_passes.push_back("conv_bn_fuse_pass");
    config.ir_passes.push_back("fc_fuse_pass");
#ifdef PADDLE_WITH_MKLDNN
    // add passes to execute with MKL-DNN
    config.ir_mkldnn_passes.push_back("conv_bn_fuse_pass");
    config.ir_mkldnn_passes.push_back("conv_eltwiseadd_bn_fuse_pass");
    config.ir_mkldnn_passes.push_back("conv_bias_mkldnn_fuse_pass");
    config.ir_mkldnn_passes.push_back("conv_elementwise_add_mkldnn_fuse_pass");
    config.ir_mkldnn_passes.push_back("conv_relu_mkldnn_fuse_pass");
    config.ir_mkldnn_passes.push_back("fc_fuse_pass");
#endif
  }
  auto predictor = CreatePaddlePredictor<contrib::AnalysisConfig,
                                         PaddleEngineKind::kAnalysis>(config);

  if (FLAGS_profile) {
    auto pf_state = paddle::platform::ProfilerState::kCPU;
    paddle::platform::EnableProfiler(pf_state);
  }

  // define output
  std::vector<PaddleTensor> output_slots;

  // run prediction
  inference::Timer timer;
  inference::Timer timer_total;
  std::vector<float> infer_accs;
  std::vector<double> batch_times;
  std::vector<double> fpses;
  for (int i = 0; i < FLAGS_iterations + FLAGS_skip_batch_num; i++) {
    if (i > 0) {
      if (!FLAGS_use_fake_data) {
        if (!reader->NextBatch(static_cast<float*>(input_data.data.data()),
                               static_cast<int64_t*>(input_label.data.data()),
                               FLAGS_batch_size, FLAGS_debug_display_images)) {
          std::cout << "No more full batches. stopping.";
          break;
        }
      }
    }

    if (FLAGS_debug_display_images)
      DataReader::drawImages(static_cast<float*>(input_data.data.data()),
                             convert_to_rgb, FLAGS_batch_size, FLAGS_channels,
                             FLAGS_width, FLAGS_height);

    if (i == FLAGS_skip_batch_num) {
      timer_total.tic();
      if (FLAGS_profile) {
        paddle::platform::ResetProfiler();
      }
    }
    std::vector<PaddleTensor> input = {input_data};
    timer.tic();
    CHECK(predictor->Run(input, &output_slots));
    double batch_time = timer.toc() / 1000;
    PADDLE_ENFORCE_EQ(output_slots.size(), 1UL);

    batch_times.push_back(batch_time);
    double fps = FLAGS_batch_size / batch_time;
    fpses.push_back(fps);
    std::string appx = (i < FLAGS_skip_batch_num) ? " (warm-up)" : "";
    printf("Iteration: %d%s, latency: %.5f s, fps: %f\n", i + 1, appx.c_str(),
           batch_time, fps);
  }

  if (FLAGS_profile) {
    paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                      "/tmp/profiler");
  }

  double total_samples = FLAGS_iterations * FLAGS_batch_size;
  double total_time = timer_total.toc() / 1000;
  PostprocessBenchmarkData(batch_times, fpses, total_time, total_samples);
}

TEST(resnet50, basic) { Main(); }

}  // namespace paddle
