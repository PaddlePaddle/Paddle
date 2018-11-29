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

//#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gflags/gflags.h>
#include <random>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/paddle_inference_api.h"
#include "paddle/fluid/platform/profiler.h"
#include "utils/image_reader.h"
#include "utils/stats.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(model_file, "__model__", "The name of a file with model.");
DEFINE_string(params_file, "",
              "The name of a single file with model parameters. By default "
              "parameters are stored in separate files.");
DEFINE_string(data_list, "", "Path to a file with a list of image files.");
DEFINE_string(data_dir, "", "Path to a directory with image files.");
DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(iterations, 1, "How many times to repeat run.");
DEFINE_int32(skip_batch_num, 0, "How many minibatches to skip in statistics.");
DEFINE_bool(use_fake_data, false, "Use fake data (1,2,...).");
DEFINE_bool(with_mkldnn, false, "Use MKL-DNN.");
DEFINE_bool(skip_passes, false, "Skip running passes.");
DEFINE_bool(enable_graphviz, false,
            "Enable graphviz to get .dot files with data flow graphs.");
DEFINE_bool(debug_display_images, false, "Show images in windows for debug.");
DEFINE_bool(with_labels, true, "The infer model do handle data labels.");
DEFINE_bool(profile, false, "Turn on profiler for fluid");
DEFINE_int32(num_threads, 1, "Number of threads for each paddle instance.");
// dimensions of imagenet images are assumed as default:
DEFINE_int32(resize_size, 256,
             "Images are resized to make smaller side have length resize_size "
             "while keeping aspect ratio.");
DEFINE_int32(crop_size, 224,
             "Resized images are cropped to crop_size x crop_size.");
DEFINE_int32(channels, 3, "Width of the image.");

namespace {
class Timer {
 public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};
}  // namespace

namespace paddle {

void PrintInfo() {
  std::cout << std::endl
            << "--- Used Parameters: -----------------" << std::endl
            << "Inference model: " << FLAGS_infer_model << std::endl
            << "Model file: " << FLAGS_model_file << std::endl
            << "Params file: " << FLAGS_params_file << std::endl
            << "File with list of images: " << FLAGS_data_list << std::endl
            << "Directory with images: " << FLAGS_data_dir << std::endl
            << "Batch size: " << FLAGS_batch_size << std::endl
            << "Number of threads: " << FLAGS_num_threads << std::endl
            << "Iterations: " << FLAGS_iterations << std::endl
            << "Number of batches to skip: " << FLAGS_skip_batch_num
            << std::endl
            << "Use fake data: " << FLAGS_use_fake_data << std::endl
            << "Use MKL-DNN: " << FLAGS_with_mkldnn << std::endl
            << "Skip passes: " << FLAGS_skip_passes << std::endl
            << "Debug display image: " << FLAGS_debug_display_images
            << std::endl
            << "Profile: " << FLAGS_profile << std::endl
            << "With labels: " << FLAGS_with_labels << std::endl
            << "Channels: " << FLAGS_channels << std::endl
            << "Resize size: " << FLAGS_resize_size << std::endl
            << "Crop size: " << FLAGS_crop_size << std::endl
            << "Enable graphviz: " << FLAGS_enable_graphviz << std::endl
            << "--------------------------------------" << std::endl;
}

// Count elements of a tensor from its shape vector
size_t Count(const std::vector<int>& shapevec) {
  auto sum = shapevec.size() > 0 ? 1 : 0;
  for (unsigned int i = 0; i < shapevec.size(); ++i) {
    sum *= shapevec[i];
  }
  return sum;
}

paddle::PaddleTensor DefineInputData() {
  std::vector<int> shape;
  shape.push_back(FLAGS_batch_size);
  shape.push_back(FLAGS_channels);
  shape.push_back(FLAGS_crop_size);
  shape.push_back(FLAGS_crop_size);
  paddle::PaddleTensor data;
  data.name = "xx";
  data.shape = shape;
  data.data.Resize(Count(shape) * sizeof(float));
  data.dtype = PaddleDType::FLOAT32;
  return data;
}

paddle::PaddleTensor DefineInputLabels() {
  paddle::PaddleTensor labels;
  labels.name = "yy";
  labels.shape = std::vector<int>({FLAGS_batch_size, 1});
  labels.data.Resize(FLAGS_batch_size * sizeof(int64_t));
  labels.dtype = paddle::PaddleDType::INT64;
  return labels;
}

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

void CreateFakeData(PaddleTensor& input_data, PaddleTensor& input_labels) {
  fill_data<float>(static_cast<float*>(input_data.data.data()),
                   Count(input_data.shape));
  int labels_size = FLAGS_batch_size;
  if (FLAGS_with_labels) {
    // create fake labels
    fill_data<int64_t>(static_cast<int64_t*>(input_labels.data.data()),
                       labels_size);
  }
}

void InitializeReader(std::unique_ptr<ImageReader>& reader,
                      bool convert_to_rgb) {
  reader.reset(new ImageReader(FLAGS_data_list, FLAGS_data_dir,
                               FLAGS_resize_size, FLAGS_crop_size,
                               FLAGS_channels, convert_to_rgb));
  if (!reader->SetSeparator('\t')) reader->SetSeparator(' ');
}

bool ReadNextBatch(PaddleTensor& input_data, PaddleTensor& input_labels,
                   std::unique_ptr<ImageReader>& reader) {
  return reader->NextBatch(static_cast<float*>(input_data.data.data()),
                           FLAGS_with_labels
                               ? static_cast<int64_t*>(input_labels.data.data())
                               : nullptr,
                           FLAGS_batch_size, FLAGS_debug_display_images);
}

void PrepareConfig(contrib::AnalysisConfig& config) {
  config.prog_file = FLAGS_infer_model + "/" + FLAGS_model_file;
  if (!FLAGS_params_file.empty()) {
    config.param_file = FLAGS_infer_model + "/" + FLAGS_params_file;
  } else {
    config.model_dir = FLAGS_infer_model;
  }
  config.use_gpu = false;
  config.device = 0;
  config.enable_ir_optim = !FLAGS_skip_passes;
  config.specify_input_name = false;
  config.SetCpuMathLibraryNumThreads(FLAGS_num_threads);
  if (FLAGS_with_mkldnn) config.EnableMKLDNN();

  // remove all passes so that we can add them in correct order
  for (int i = config.pass_builder()->AllPasses().size() - 1; i >= 0; i--)
    config.pass_builder()->DeletePass(i);

  // add passes
  config.pass_builder()->AppendPass("infer_clean_graph_pass");
  if (FLAGS_with_mkldnn) {
    // add passes to execute with MKL-DNN
    config.pass_builder()->AppendPass("mkldnn_placement_pass");
    config.pass_builder()->AppendPass("depthwise_conv_mkldnn_pass");
    config.pass_builder()->AppendPass("conv_bn_fuse_pass");
    config.pass_builder()->AppendPass("conv_eltwiseadd_bn_fuse_pass");
    config.pass_builder()->AppendPass("conv_bias_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("conv_elementwise_add_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("conv_relu_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("fc_fuse_pass");
    config.pass_builder()->AppendPass("is_test_pass");
  } else {
    // add passes to execute keeping the order - without MKL-DNN
    config.pass_builder()->AppendPass("conv_bn_fuse_pass");
    config.pass_builder()->AppendPass("fc_fuse_pass");
  }

  if (FLAGS_enable_graphviz) config.pass_builder()->TurnOnDebug();
}

void Main() {
  PrintInfo();

  if (FLAGS_batch_size <= 0)
    throw std::invalid_argument(
        "The batch_size option is less than or equal to 0.");
  if (FLAGS_iterations <= 0)
    throw std::invalid_argument(
        "The iterations option is less than or equal to 0.");
  if (FLAGS_skip_batch_num < 0)
    throw std::invalid_argument("The skip_batch_num option is less than 0.");
  struct stat sb;
  if (stat(FLAGS_infer_model.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
    throw std::invalid_argument(
        "The inference model directory does not exist.");
  }
  if (FLAGS_with_labels && FLAGS_use_fake_data)
    throw std::invalid_argument("Cannot use fake data for accuracy measuring.");

  paddle::PaddleTensor input_data = DefineInputData();
  paddle::PaddleTensor input_labels = DefineInputLabels();

  // reader instance for real data
  std::unique_ptr<ImageReader> reader;
  bool convert_to_rgb = true;

  // read the first batch
  if (FLAGS_use_fake_data) {
    CreateFakeData(input_data, input_labels);
  } else {
    InitializeReader(reader, convert_to_rgb);
    if (!ReadNextBatch(input_data, input_labels, reader)) {
      throw std::invalid_argument(
          "Batch size cannot be bigger than dataset size.");
    }
  }

  // configure predictor
  contrib::AnalysisConfig config;
  PrepareConfig(config);

  auto predictor = CreatePaddlePredictor<contrib::AnalysisConfig,
                                         PaddleEngineKind::kAnalysis>(config);

  if (FLAGS_profile) {
    auto pf_state = paddle::platform::ProfilerState::kCPU;
    paddle::platform::EnableProfiler(pf_state);
  }

  // define output
  std::vector<PaddleTensor> output_slots;

  // run prediction
  Timer timer;
  Timer timer_total;
  Stats stats(FLAGS_batch_size, FLAGS_skip_batch_num);
  for (int i = 0; i < FLAGS_iterations + FLAGS_skip_batch_num; i++) {
    // start gathering performance data after `skip_batch_num` iterations
    if (i == FLAGS_skip_batch_num) {
      timer_total.tic();
      if (FLAGS_profile) {
        paddle::platform::ResetProfiler();
      }
    }

    // read next batch of data
    if (i > 0 && !FLAGS_use_fake_data) {
      if (!ReadNextBatch(input_data, input_labels, reader)) {
        std::cout << "No more full batches. Stopping." << std::endl;
        break;
      }
    }

    // display images from batch if requested
    if (FLAGS_debug_display_images)
      ImageReader::drawImages(static_cast<float*>(input_data.data.data()),
                              convert_to_rgb, FLAGS_batch_size, FLAGS_channels,
                              FLAGS_crop_size, FLAGS_crop_size);

    // define input
    std::vector<PaddleTensor> input;
    if (FLAGS_with_labels)
      input = {input_data, input_labels};
    else
      input = {input_data};

    // run inference
    timer.tic();
    if (!predictor->Run(input, &output_slots))
      throw std::runtime_error("Prediction failed.");
    double batch_time = timer.toc() / 1000;

    // gather statistics from the iteration
    stats.Gather(output_slots, batch_time, i);
  }

  if (FLAGS_profile) {
    paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                      "/tmp/profiler");
  }

  // postprocess statistics from all iterations
  double total_samples = FLAGS_iterations * FLAGS_batch_size;
  double total_time = timer_total.toc() / 1000;
  stats.Postprocess(total_time, total_samples);
}

}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    paddle::Main();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return 0;
}
