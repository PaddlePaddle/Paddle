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

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

class TopKOpConverter : public OpConverter {
 public:
  TopKOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr);

    auto* input_tensor = engine_->GetITensor(op_desc.Input("X")[0]);

    const int k = op_desc.HasAttr("k")
                      ? PADDLE_GET_CONST(int, op_desc.GetAttr("k"))
                      : 1.0f;

    nvinfer1::Dims input_dims = input_tensor->getDimensions();
    int axis = input_dims.nbDims;
    nvinfer1::ITopKLayer* layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             TopK,
                             *input_tensor,
                             nvinfer1::TopKOperation::kMAX,
                             k,
                             1 << (axis - 1));

    std::vector<std::string> output_names;
    output_names.push_back(op_desc.Output("Out").front());
    output_names.push_back(op_desc.Output("Indices").front());

    RreplenishLayerAndOutput(layer, "top_k", output_names, test_mode);
  }
};
class TopKv2OpConverter : public OpConverter {
 public:
  TopKv2OpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr);

    auto* input_tensor = engine_->GetITensor(op_desc.Input("X")[0]);

    const int k = op_desc.HasAttr("k")
                      ? PADDLE_GET_CONST(int, op_desc.GetAttr("k"))
                      : 1.0f;
    const int axis = op_desc.HasAttr("axis")
                         ? PADDLE_GET_CONST(int, op_desc.GetAttr("axis"))
                         : 1.0f;
    const bool largest =
        op_desc.HasAttr("largest")
            ? PADDLE_GET_CONST(bool, op_desc.GetAttr("largest"))
            : true;
    auto flag =
        largest ? nvinfer1::TopKOperation::kMAX : nvinfer1::TopKOperation::kMIN;
    nvinfer1::ITopKLayer* layer = nullptr;
    if (axis == -1) {
      nvinfer1::Dims input_dims = input_tensor->getDimensions();
      layer = TRT_ENGINE_ADD_LAYER(
          engine_, TopK, *input_tensor, flag, k, 1 << (input_dims.nbDims - 1));
    } else {
      if (engine_->with_dynamic_shape()) {
        layer = TRT_ENGINE_ADD_LAYER(
            engine_, TopK, *input_tensor, flag, k, 1 << axis);
      } else {
        layer = TRT_ENGINE_ADD_LAYER(
            engine_, TopK, *input_tensor, flag, k, 1 << (axis - 1));
      }
    }
    std::vector<std::string> output_names;
    output_names.push_back(op_desc.Output("Out").front());
    output_names.push_back(op_desc.Output("Indices").front());

    RreplenishLayerAndOutput(layer, "top_k_v2", output_names, test_mode);
  }
};
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(top_k, TopKOpConverter);
REGISTER_TRT_OP_CONVERTER(top_k_v2, TopKv2OpConverter);
