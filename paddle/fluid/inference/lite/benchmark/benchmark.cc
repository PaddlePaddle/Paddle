/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <unistd.h>
#include <fstream>
#include <sstream>
#include <vector>
#include "paddle/fluid/inference/lite/anakin_config.h"
#include "paddle/fluid/inference/lite/paddle_api.h"

namespace paddle {

void PrintShape(const std::vector<int> &shape) {
  std::ostringstream os;
  os << "Shape: ";
  if (shape.size() > 0) {
    os << shape[0];
    for (int i = 1; i < shape.size(); ++i) {
      os << ", " << shape[i];
    }
  }
  LOG(INFO) << os.str();
}

int ShapeSize(const std::vector<int> &shape) {
  int size = 1;
  for (int j = 0; j < shape.size(); ++j) {
    size *= shape[j];
  }
  return size;
}

template <typename T>
int InitTensorValFromFile(const std::string &file, PaddleTensor *tensor) {
  int size = ShapeSize(tensor->shape);
  void *tensor_data = tensor->data.data();
  std::ifstream in(file, std::ios::in | std::ios::binary);
  in.read(reinterpret_cast<char *>(tensor_data), size * sizeof(T));
  in.close();
}

int SetupTensors(const std::vector<std::vector<int>> &shapes,
                 const std::vector<std::string> &names,
                 std::vector<PaddleTensor> *outputs) {
  while (outputs->size() < shapes.size()) {
    outputs->emplace_back();
  }
  for (int i = 0; i < shapes.size(); ++i) {
    int size = ShapeSize(shapes[i]);
    outputs->at(i).name = names[i];
    outputs->at(i).shape = shapes[i];
    outputs->at(i).data.Resize(size * sizeof(float));
    outputs->at(i).dtype = FLOAT32;
  }
}

int test(const char *model, const char *image, const char *image_shape,
         const int quant, const int times) {
  contrib::AnakinConfig config;
  config.model_file = std::string(model);
  // config.model_file = "./mobilenetv1.anakin.bin";
  config.max_batch_size = 1;
  config.precision_type =
      (quant == 1) ? contrib::AnakinConfig::INT8 : contrib::AnakinConfig::FP32;

  LOG(INFO) << "quant: " << quant;

  std::unique_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<contrib::AnakinConfig, PaddleEngineKind::kAnakin>(
          config);

  LOG(INFO) << "create predictor success";
  std::vector<std::string> in_names = predictor->GetInputNames();
  std::vector<PaddleTensor> inputs, outpus;
  std::vector<std::vector<int>> in_shapes;
  std::vector<int> dim{1, 3, 224, 224};
  sscanf(image_shape, "%d,%d,%d,%d", &dim[0], &dim[1], &dim[2], &dim[3]);
  in_shapes.push_back(dim);
  SetupTensors(in_shapes, in_names, &inputs);
  PrintShape(dim);

  // InitTensorValFromFile<float>("./test_image_1x3x224x224_float", &inputs[0]);
  InitTensorValFromFile<float>(std::string(image), &inputs[0]);
  LOG(INFO) << "init tensor value success";

  std::vector<std::string> out_names = predictor->GetOutputNames();
  LOG(INFO) << "output size: " << out_names.size();
  outpus.resize(out_names.size());
  for (int i = 0; i < out_names.size(); ++i) {
    outpus[i].name = out_names[i];
  }

  LOG(INFO) << "start run prediction";
  predictor->Run(inputs, &outpus);

  struct timespec ts_begin, ts_end;
  clock_gettime(CLOCK_MONOTONIC, &ts_begin);
  for (int i = 0; i < times; ++i) {
    predictor->Run(inputs, &outpus);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  uint64_t elapsed = (ts_end.tv_sec - ts_begin.tv_sec) * 1e3 +
                     (ts_end.tv_nsec - ts_begin.tv_nsec) / 1e6;
  LOG(INFO) << "elapsed: " << (1.f * elapsed) / times << " ms";

  LOG(INFO) << "finish prediction";

  for (int i = 0; i < outpus.size(); ++i) {
    int size = ShapeSize(outpus[i].shape);
    // int stride = (size + 19) / 20;
    int stride = 1;
    int loop = size / stride;
    float *output_data = static_cast<float *>(outpus[i].data.data());
    std::ostringstream os;
    os << output_data[0];
    for (int j = 1; j < loop; ++j) {
      os << ", " << output_data[j * stride];
    }
    LOG(INFO) << os.str();
  }
  return 0;
}

}  // namespace paddle

int main(int argc, char *argv[]) {
  if (argc < 6) {
    LOG(INFO) << "Usage: ./benchmark [model] [image] [image-shape] [8bit] "
                 "[run-times]";
    LOG(INFO) << "Example:";
    LOG(INFO) << "    ./benchmark ./mobilenetv1.model ./test_image.bin "
                 "1,3,224,224 0 10";
    return 1;
  }
  int quant_8bit = atoi(argv[4]);
  int times = atoi(argv[5]);
  return paddle::test(argv[1], argv[2], argv[3], quant_8bit, times);
}
