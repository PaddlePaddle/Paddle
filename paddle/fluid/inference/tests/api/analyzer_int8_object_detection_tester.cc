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

DEFINE_bool(enable_mkldnn, true, "Enable MKLDNN");

// setting iterations to 0 means processing the whole dataset
namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchIrOptim(true);
  cfg->SwitchSpecifyInputNames(false);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
  if (FLAGS_enable_mkldnn) cfg->EnableMKLDNN();
}

std::vector<size_t> ReadObjectsNum(std::ifstream &file, size_t offset,
                                   int64_t total_images) {
  std::vector<size_t> num_objects;
  num_objects.resize(total_images);

  file.clear();
  file.seekg(offset);
  file.read(reinterpret_cast<char *>(num_objects.data()),
            total_images * sizeof(size_t));

  if (file.eof()) LOG(ERROR) << "Reached end of stream";
  if (file.fail()) throw std::runtime_error("Failed reading file.");
  return num_objects;
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset, std::string name)
      : file_(file), position_(beginning_offset), name_(name) {}

  PaddleTensor NextBatch(std::vector<int> shape, std::vector<size_t> lod) {
    int numel =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel * sizeof(T));
    if (lod.empty() == false) {
      tensor.lod.clear();
      tensor.lod.push_back(lod);
    }
    file_.seekg(position_);
    file_.read(reinterpret_cast<char *>(tensor.data.data()), numel * sizeof(T));
    position_ = file_.tellg();
    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");
    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position_;
  std::string name_;
};

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    FAIL() << "Couldn't open file: " << FLAGS_infer_data;
  }

  int64_t total_images{0};
  file.read(reinterpret_cast<char *>(&total_images), sizeof(int64_t));
  LOG(INFO) << "Total images in file: " << total_images;

  size_t image_beginning_offset = static_cast<size_t>(file.tellg());
  auto lod_offset_in_file =
      image_beginning_offset + sizeof(float) * total_images * 3 * 300 * 300;
  auto labels_beginning_offset =
      lod_offset_in_file + sizeof(size_t) * total_images;

  std::vector<size_t> lod_full =
      ReadObjectsNum(file, lod_offset_in_file, total_images);
  size_t sum_objects_num =
      std::accumulate(lod_full.begin(), lod_full.end(), 0UL);

  auto bbox_beginning_offset =
      labels_beginning_offset + sizeof(int64_t) * sum_objects_num;
  auto difficult_beginning_offset =
      bbox_beginning_offset + sizeof(float) * sum_objects_num * 4;

  TensorReader<float> image_reader(file, image_beginning_offset, "image");
  TensorReader<int64_t> label_reader(file, labels_beginning_offset, "gt_label");
  TensorReader<float> bbox_reader(file, bbox_beginning_offset, "gt_bbox");
  TensorReader<int64_t> difficult_reader(file, difficult_beginning_offset,
                                         "gt_difficult");
  auto iterations_max = total_images / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }
  for (auto i = 0; i < iterations; i++) {
    auto images_tensor = image_reader.NextBatch({batch_size, 3, 300, 300}, {});
    std::vector<size_t> batch_lod(lod_full.begin() + i * batch_size,
                                  lod_full.begin() + batch_size * (i + 1));
    size_t batch_num_objects =
        std::accumulate(batch_lod.begin(), batch_lod.end(), 0UL);
    batch_lod.insert(batch_lod.begin(), 0UL);
    for (auto it = batch_lod.begin() + 1; it != batch_lod.end(); it++) {
      *it = *it + *(it - 1);
    }
    auto labels_tensor = label_reader.NextBatch(
        {static_cast<int>(batch_num_objects), 1}, batch_lod);
    auto bbox_tensor = bbox_reader.NextBatch(
        {static_cast<int>(batch_num_objects), 4}, batch_lod);
    auto difficult_tensor = difficult_reader.NextBatch(
        {static_cast<int>(batch_num_objects), 1}, batch_lod);

    inputs->emplace_back(std::vector<PaddleTensor>{
        std::move(images_tensor), std::move(bbox_tensor),
        std::move(labels_tensor), std::move(difficult_tensor)});
  }
}

std::shared_ptr<std::vector<PaddleTensor>> GetWarmupData(
    const std::vector<std::vector<PaddleTensor>> &test_data,
    int32_t num_images = FLAGS_warmup_batch_size) {
  int test_data_batch_size = test_data[0][0].shape[0];
  auto iterations = test_data.size();
  PADDLE_ENFORCE_LE(
      static_cast<size_t>(num_images), iterations * test_data_batch_size,
      paddle::platform::errors::Fatal(
          "The requested quantization warmup data size " +
          std::to_string(num_images) + " is bigger than all test data size."));

  PaddleTensor images;
  images.name = "image";
  images.shape = {num_images, 3, 300, 300};
  images.dtype = PaddleDType::FLOAT32;
  images.data.Resize(sizeof(float) * num_images * 3 * 300 * 300);

  int batches = num_images / test_data_batch_size;
  int batch_remain = num_images % test_data_batch_size;
  size_t num_objects = 0UL;
  std::vector<size_t> accum_lod;
  accum_lod.push_back(0UL);
  for (int i = 0; i < batches; i++) {
    std::transform(test_data[i][1].lod[0].begin() + 1,
                   test_data[i][1].lod[0].end(), std::back_inserter(accum_lod),
                   [&num_objects](size_t lodtemp) -> size_t {
                     return lodtemp + num_objects;
                   });
    num_objects += test_data[i][1].lod[0][test_data_batch_size];
  }
  if (batch_remain > 0) {
    std::transform(test_data[batches][1].lod[0].begin() + 1,
                   test_data[batches][1].lod[0].begin() + batch_remain + 1,
                   std::back_inserter(accum_lod),
                   [&num_objects](size_t lodtemp) -> size_t {
                     return lodtemp + num_objects;
                   });
    num_objects = num_objects + test_data[batches][1].lod[0][batch_remain];
  }

  PaddleTensor labels;
  labels.name = "gt_label";
  labels.shape = {static_cast<int>(num_objects), 1};
  labels.dtype = PaddleDType::INT64;
  labels.data.Resize(sizeof(int64_t) * num_objects);
  labels.lod.push_back(accum_lod);

  PaddleTensor bbox;
  bbox.name = "gt_bbox";
  bbox.shape = {static_cast<int>(num_objects), 4};
  bbox.dtype = PaddleDType::FLOAT32;
  bbox.data.Resize(sizeof(float) * num_objects * 4);
  bbox.lod.push_back(accum_lod);

  PaddleTensor difficult;
  difficult.name = "gt_difficult";
  difficult.shape = {static_cast<int>(num_objects), 1};
  difficult.dtype = PaddleDType::INT64;
  difficult.data.Resize(sizeof(int64_t) * num_objects);
  difficult.lod.push_back(accum_lod);

  size_t objects_accum = 0;
  size_t objects_in_batch = 0;
  for (int i = 0; i < batches; i++) {
    objects_in_batch = test_data[i][1].lod[0][test_data_batch_size];
    std::copy_n(static_cast<float *>(test_data[i][0].data.data()),
                test_data_batch_size * 3 * 300 * 300,
                static_cast<float *>(images.data.data()) +
                    i * test_data_batch_size * 3 * 300 * 300);
    std::copy_n(static_cast<int64_t *>(test_data[i][1].data.data()),
                objects_in_batch,
                static_cast<int64_t *>(labels.data.data()) + objects_accum);
    std::copy_n(static_cast<float *>(test_data[i][2].data.data()),
                objects_in_batch * 4,
                static_cast<float *>(bbox.data.data()) + objects_accum * 4);
    std::copy_n(static_cast<int64_t *>(test_data[i][3].data.data()),
                objects_in_batch,
                static_cast<int64_t *>(difficult.data.data()) + objects_accum);
    objects_accum = objects_accum + objects_in_batch;
  }
  if (batch_remain > 0) {
    size_t objects_remain = test_data[batches][1].lod[0][batch_remain];
    std::copy_n(static_cast<float *>(test_data[batches][0].data.data()),
                batch_remain * 3 * 300 * 300,
                static_cast<float *>(images.data.data()) +
                    objects_accum * 3 * 300 * 300);
    std::copy_n(static_cast<int64_t *>(test_data[batches][1].data.data()),
                objects_remain,
                static_cast<int64_t *>(labels.data.data()) + objects_accum);
    std::copy_n(static_cast<float *>(test_data[batches][2].data.data()),
                objects_remain * 4,
                static_cast<float *>(bbox.data.data()) + objects_accum * 4);
    std::copy_n(static_cast<int64_t *>(test_data[batches][3].data.data()),
                objects_remain,
                static_cast<int64_t *>(difficult.data.data()) + objects_accum);
    objects_accum = objects_accum + objects_remain;
  }
  PADDLE_ENFORCE_EQ(
      static_cast<size_t>(num_objects), static_cast<size_t>(objects_accum),
      paddle::platform::errors::Fatal("The requested num of objects " +
                                      std::to_string(num_objects) +
                                      " is the same as objects_accum."));

  auto warmup_data = std::make_shared<std::vector<PaddleTensor>>(4);
  (*warmup_data)[0] = std::move(images);
  (*warmup_data)[1] = std::move(bbox);
  (*warmup_data)[2] = std::move(labels);
  (*warmup_data)[3] = std::move(difficult);

  return warmup_data;
}

TEST(Analyzer_int8_mobilenet_ssd, quantization) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  AnalysisConfig q_cfg;
  SetConfig(&q_cfg);

  // read data from file and prepare batches with test data
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);

  // prepare warmup batch from input data read earlier
  // warmup batch size can be different than batch size
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data =
      GetWarmupData(input_slots_all);

  // configure quantizer
  if (FLAGS_enable_mkldnn) {
    q_cfg.EnableMkldnnQuantizer();
    q_cfg.mkldnn_quantizer_config();
    std::unordered_set<std::string> quantize_operators(
        {"conv2d", "depthwise_conv2d", "prior_box", "transpose2", "reshape2"});
    q_cfg.mkldnn_quantizer_config()->SetEnabledOpTypes(quantize_operators);
    q_cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data);
    q_cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(
        FLAGS_warmup_batch_size);
  }

  // 0 is avg_cost, 1 is top1_acc, 2 is top5_acc or mAP
  CompareQuantizedAndAnalysis(&cfg, &q_cfg, input_slots_all, 2);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
