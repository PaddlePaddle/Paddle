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
#include <opencv2/opencv.hpp>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/helper.h"
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
DEFINE_bool(use_fake_data, false, "Use fake data (1,2,...).");
DEFINE_bool(skip_passes, false, "Skip running passes.");
DEFINE_bool(debug_display_images, false, "Show images in windows for debug.");

namespace paddle {

struct DataReader {
  explicit DataReader(const std::string& path)
      : file(new std::ifstream(path)) {}

  bool NextBatch(float* input, int64_t* label) {
    std::string line;

    if (!file->is_open()) {
      throw std::invalid_argument("Cannot open FLAGS_data_list file " +
                                  FLAGS_data_list);
    }
    if (FLAGS_data_dir.empty()) {
      throw std::invalid_argument(
          "FLAGS_data_dir must be set to use imagenet.");
    }

    for (int i = 0; i < FLAGS_batch_size; i++) {
      if (!std::getline(*file, line)) return false;

      std::vector<std::string> pieces;
      inference::split(line, '\t', &pieces);
      auto filename = FLAGS_data_dir + pieces.at(0);
      label[i] = std::stoi(pieces.at(1));

      cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
      if (image.data == nullptr) {
        std::string error_msg = "Couldn't open file " + filename;
        throw std::runtime_error(error_msg);
      }
      if (FLAGS_debug_display_images)
        cv::imshow(std::to_string(i) + " input image", image);
      cv::Mat image2;
      cv::resize(image, image2, cv::Size(FLAGS_width, FLAGS_height));

      cv::Mat fimage;
      image2.convertTo(fimage, CV_32FC3);

      fimage /= 255.f;
      cv::Scalar mean(0.406f, 0.456f, 0.485f);
      cv::Scalar std(0.225f, 0.224f, 0.229f);

      std::vector<cv::Mat> fimage_channels;
      cv::split(fimage, fimage_channels);

      for (int c = 0; c < FLAGS_channels; c++) {
        fimage_channels[c] -= mean[c];
        fimage_channels[c] /= std[c];
        for (int row = 0; row < fimage.rows; ++row) {
          const float* fimage_begin = fimage_channels[c].ptr<const float>(row);
          const float* fimage_end = fimage_begin + fimage.cols;
          std::copy(fimage_begin, fimage_end,
                    input + row * fimage.cols + c * fimage.cols * fimage.rows +
                        i * 3 * fimage.cols * fimage.rows);
        }
      }
    }
    return true;
  }

  std::unique_ptr<std::ifstream> file;
};

template <typename T>
void fill_data(T* data, unsigned int count) {
  for (unsigned int i = 0; i < count; ++i) {
    *(data + i) = i;
  }
}

void PrintResults(int bs, int iterations, double lat_avg, float acc_avg) {
  LOG(INFO) << "===========profile result===========";
  LOG(INFO) << "batch_size: " << bs << ", iterations: " << iterations
            << ", avg latency: " << lat_avg << "ms"
            << ", avg fps: " << bs * 1000 / lat_avg
            << ", avg accuracy: " << acc_avg;
  LOG(INFO) << "=====================================";
}

void Main(int batch_size) {
  auto count = [](std::vector<int>& shapevec) {
    auto sum = shapevec.size() > 0 ? 1 : 0;
    for (unsigned int i = 0; i < shapevec.size(); ++i) {
      sum *= shapevec[i];
    }
    return sum;
  };

  // define input: input
  std::vector<int> shape;
  shape.push_back(FLAGS_batch_size);
  shape.push_back(FLAGS_channels);
  shape.push_back(FLAGS_height);
  shape.push_back(FLAGS_width);
  paddle::PaddleTensor input;
  input.name = "xx";
  input.shape = shape;

  // define input: label
  int label_size = FLAGS_batch_size;
  paddle::PaddleTensor input_label;
  input_label.data.Resize(label_size * sizeof(int64_t));
  input_label.name = "yy";
  input_label.shape = std::vector<int>({label_size, 1});
  input_label.dtype = paddle::PaddleDType::INT64;

  if (FLAGS_use_fake_data) {
    // create fake data
    input.data.Resize(count(shape) * sizeof(float));
    fill_data<float>(static_cast<float*>(input.data.data()), count(shape));

    input.dtype = paddle::PaddleDType::FLOAT32;

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
    // get imagenet data and label
    input.data.Resize(count(shape) * sizeof(float));
    input.dtype = PaddleDType::FLOAT32;

    DataReader reader(FLAGS_data_list);

    reader.NextBatch(static_cast<float*>(input.data.data()),
                     static_cast<int64_t*>(input_label.data.data()));
  }

  if (FLAGS_debug_display_images) {
    for (int b = 0; b < FLAGS_batch_size; b++) {
      std::vector<cv::Mat> fimage_channels;
      for (int c = 0; c < FLAGS_channels; c++) {
        fimage_channels.emplace_back(
            cv::Size(FLAGS_width, FLAGS_height), CV_32FC1,
            static_cast<float*>(input.data.data()) +
                FLAGS_width * FLAGS_height * c +
                FLAGS_width * FLAGS_height * FLAGS_channels * b);
      }
      cv::Mat mat;
      cv::merge(fimage_channels, mat);
      cv::imshow(std::to_string(b) + " output image", mat);
    }
    cv::waitKey(0);
  }

  // create predictor
  contrib::AnalysisConfig config;
  // MKLDNNAnalysisConfig config;
  config.model_dir = FLAGS_infer_model;
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
  CHECK_GE(output_slots.size(), 1UL);
  PaddleTensor output = output_slots[0];
  CHECK_EQ(output.lod.size(), 0UL);
  CHECK_EQ(output.dtype, paddle::PaddleDType::FLOAT32);
  float* odata = static_cast<float*>(output.data.data());
  size_t olen = output.data.length() / sizeof(FLOAT32);
  float acc_avg = std::accumulate(odata, odata + olen, 0.0) / olen;
  double lat_avg = sum / FLAGS_iterations;

  PrintResults(batch_size, FLAGS_iterations, lat_avg, acc_avg);
}

TEST(resnet50, basic) { Main(FLAGS_batch_size); }

}  // namespace paddle
