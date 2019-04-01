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
#include "paddle/fluid/inference/tests/api/tester_helper.h"

DEFINE_int32(iterations, 0, "Number of iterations");

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->SetProgFile("__model__");
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames(false);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);

  cfg->EnableMKLDNN();
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset,
               std::vector<int> shape, std::string name)
      : file_(file), position(beginning_offset), shape_(shape), name_(name) {
    numel =
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<T>());
  }

  PaddleTensor NextBatch() {
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape_;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel * sizeof(T));

    file_.seekg(position);
    file_.read(static_cast<char *>(tensor.data.data()), numel * sizeof(T));
    position = file_.tellg();

    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");

    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position;
  std::vector<int> shape_;
  std::string name_;
  size_t numel;
};

std::shared_ptr<std::vector<PaddleTensor>> GetWarmupData(
    const std::vector<std::vector<PaddleTensor>> &test_data, int num_images) {
  int test_data_batch_size = test_data[0][0].shape[0];
  CHECK_LE(static_cast<size_t>(num_images),
           test_data.size() * test_data_batch_size);

  PaddleTensor images;
  images.name = "input";
  images.shape = {num_images, 3, 224, 224};
  images.dtype = PaddleDType::FLOAT32;
  images.data.Resize(sizeof(float) * num_images * 3 * 224 * 224);

  PaddleTensor labels;
  labels.name = "labels";
  labels.shape = {num_images, 1};
  labels.dtype = PaddleDType::INT64;
  labels.data.Resize(sizeof(int64_t) * num_images);

  for (int i = 0; i < num_images; i++) {
    auto batch = i / test_data_batch_size;
    auto element_in_batch = i % test_data_batch_size;
    std::copy_n(static_cast<float *>(test_data[batch][0].data.data()) +
                    element_in_batch * 3 * 224 * 224,
                3 * 224 * 224,
                static_cast<float *>(images.data.data()) + i * 3 * 224 * 224);

    std::copy_n(static_cast<int64_t *>(test_data[batch][1].data.data()) +
                    element_in_batch,
                1, static_cast<int64_t *>(labels.data.data()) + i);
  }

  auto warmup_data = std::make_shared<std::vector<PaddleTensor>>(2);
  (*warmup_data)[0] = std::move(images);
  (*warmup_data)[1] = std::move(labels);
  return warmup_data;
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
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
  auto labels_offset_in_file =
      static_cast<size_t>(file.tellg()) +
      sizeof(float) * total_images *
          std::accumulate(image_batch_shape.begin() + 1,
                          image_batch_shape.end(), 1, std::multiplies<int>());

  TensorReader<float> image_reader(file, 0, image_batch_shape, "input");
  TensorReader<int64_t> label_reader(file, labels_offset_in_file,
                                     label_batch_shape, "label");

  auto iterations = total_images / batch_size;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations)
    iterations = FLAGS_iterations;
  for (auto i = 0; i < iterations; i++) {
    auto images = image_reader.NextBatch();
    auto labels = label_reader.NextBatch();
    inputs->emplace_back(
        std::vector<PaddleTensor>{std::move(images), std::move(labels)});
  }
}

TEST(Analyzer_int8_resnet50, quantization) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  AnalysisConfig q_cfg;
  SetConfig(&q_cfg);

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all, 100);

  std::shared_ptr<std::vector<PaddleTensor>> warmup_data =
      GetWarmupData(input_slots_all, 100);

  q_cfg.EnableMkldnnQuantizer();
  q_cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data);
  q_cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(100);

  CompareQuantizedAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
      reinterpret_cast<const PaddlePredictor::Config *>(&q_cfg),
      input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
