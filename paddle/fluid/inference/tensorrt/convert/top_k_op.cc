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

#include <NvInfer.h>

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::inference::tensorrt {

class TopKOpConverter : public OpConverter {
 public:
  TopKOpConverter() = default;
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a top_k op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);

    auto* input_tensor = engine_->GetITensor(op_desc.Input("X")[0]);

    const int k =
        op_desc.HasAttr("k") ? PADDLE_GET_CONST(int, op_desc.GetAttr("k")) : 1;
    int axis = op_desc.HasAttr("axis")
                   ? PADDLE_GET_CONST(int, op_desc.GetAttr("axis"))
                   : -1;
    const bool largest =
        op_desc.HasAttr("largest")
            ? PADDLE_GET_CONST(bool, op_desc.GetAttr("largest"))
            : true;
    auto flag =
        largest ? nvinfer1::TopKOperation::kMAX : nvinfer1::TopKOperation::kMIN;

    auto input_dims = input_tensor->getDimensions();
    auto input_rank = input_dims.nbDims;
    // 1d needs expand to 2d
    bool expand_to_2d = (input_rank == 1);
    if (expand_to_2d) {
      input_tensor = Unsqueeze(input_tensor, std::vector<int32_t>{1});
    }

    // INT32 only, other data type should to casted to INT32.
    nvinfer1::DataType type = input_tensor->getType();
    bool cast = (type == nvinfer1::DataType::kINT32);
    if (cast) {
      input_tensor = Cast(input_tensor, nvinfer1::DataType::kFLOAT);
    }

    nvinfer1::ITopKLayer* layer = nullptr;
    if (axis < 0) axis += input_rank;

    layer =
        TRT_ENGINE_ADD_LAYER(engine_, TopK, *input_tensor, flag, k, 1 << axis);

    nvinfer1::ITensor* values = layer->getOutput(0);
    nvinfer1::ITensor* indices = layer->getOutput(1);

    // un-expand to 1d
    if (expand_to_2d) {
      values = Squeeze(values, std::vector<int32_t>{1});
      indices = Squeeze(indices, std::vector<int32_t>{1});
    }

    // cast back
    if (cast) {
      values = Cast(values, nvinfer1::DataType::kINT32);
    }

    auto out_name = op_desc.Output("Out").front();
    auto indices_name = op_desc.Output("Indices").front();
    values->setName(out_name.c_str());
    engine_->SetITensor(out_name.c_str(), values);

    indices->setName(indices_name.c_str());
    engine_->SetITensor(indices_name.c_str(), indices);

    layer->setName(
        ("top_k (Output: " + out_name + "," + indices_name + ")").c_str());
  }
};
}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(top_k, TopKOpConverter);
REGISTER_TRT_OP_CONVERTER(top_k_v2, TopKOpConverter);
