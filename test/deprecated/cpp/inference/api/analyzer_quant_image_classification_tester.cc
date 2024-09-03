/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>

#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "test/cpp/inference/api/tester_helper.h"

PD_DEFINE_bool(enable_mkldnn, true, "Enable MKLDNN");

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg, std::string model_path) {
  cfg->SetModel(model_path);
  cfg->DisableGpu();
  cfg->SwitchIrOptim(true);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
  cfg->EnableNewIR();
  cfg->EnableNewExecutor();
  cfg->SetOptimizationLevel(3);

  if (FLAGS_enable_mkldnn) cfg->EnableMKLDNN();
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file,
               size_t beginning_offset,
               std::vector<int> shape,
               std::string name)
      : file_(file),
        position_(beginning_offset),
        shape_(shape),
        name_(name),
        numel_(0) {
    numel_ = std::accumulate(
        shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());
  }

  PaddleTensor NextBatch() {
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape_;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel_ * sizeof(T));

    file_.seekg(position_);
    file_.read(static_cast<char *>(tensor.data.data()), numel_ * sizeof(T));
    position_ = file_.tellg();

    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");

    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position_;
  std::vector<int> shape_;
  std::string name_;
  size_t numel_;
};

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              bool with_accuracy_layer = FLAGS_with_accuracy_layer,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    FAIL() << "Couldn't open file: " << FLAGS_infer_data;
  }

  int64_t total_images{0};
  file.read(reinterpret_cast<char *>(&total_images), sizeof(total_images));
  LOG(INFO) << "Total images in file: " << total_images;

  std::vector<int> image_batch_shape{batch_size, 3, 224, 224};
  std::vector<int> label_batch_shape{batch_size, 1};
  auto images_offset_in_file = static_cast<size_t>(file.tellg());

  TensorReader<float> image_reader(
      file, images_offset_in_file, image_batch_shape, "image");

  auto iterations_max = total_images / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }

  auto labels_offset_in_file =
      images_offset_in_file + sizeof(float) * total_images * 3 * 224 * 224;

  TensorReader<int64_t> label_reader(
      file, labels_offset_in_file, label_batch_shape, "label");
  for (auto i = 0; i < iterations; i++) {
    auto images = image_reader.NextBatch();
    std::vector<PaddleTensor> tmp_vec;
    tmp_vec.push_back(std::move(images));
    if (with_accuracy_layer) {
      auto labels = label_reader.NextBatch();
      tmp_vec.push_back(std::move(labels));
    }
    inputs->push_back(std::move(tmp_vec));
  }
}

TEST(Analyzer_quant_image_classification, quantization) {
  AnalysisConfig fp32_cfg;
  SetConfig(&fp32_cfg, FLAGS_fp32_model);
  fp32_cfg.EnableMKLDNN();

  AnalysisConfig int8_cfg;
  SetConfig(&int8_cfg, FLAGS_int8_model);
  if (FLAGS_enable_int8_qat) int8_cfg.EnableMkldnnInt8();

  // read data from file and prepare batches with test data
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);

  // 0 is avg_cost, 1 is top1_accuracy, 2 is top5_accuracy or mAP
  CompareAnalysisAndAnalysis(
      &fp32_cfg, &int8_cfg, input_slots_all, FLAGS_with_accuracy_layer, 1);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
