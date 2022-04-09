/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

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

/*
 * Stack converter from fluid to tensorRT.
 */
class StridedSliceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid StridedSlice op to tensorrt Slice layer";

    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    nvinfer1::Dims input_dims = input->getDimensions();

    std::vector<int> axes =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    std::vector<int> starts =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("starts"));
    std::vector<int> ends =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("ends"));
    std::vector<int> strides =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));

    nvinfer1::Dims start;
    start.nbDims = input_dims.nbDims;
    int axes_size = axes.size();
    for (int i = 0; i < start.nbDims; i++) {
      start.d[i] = 0;
    }
    for (int i = 0; i < axes_size; i++) {
      start.d[axes[i]] = starts[i];
    }

    nvinfer1::Dims stride;
    stride.nbDims = input_dims.nbDims;
    for (int i = 0; i < stride.nbDims; i++) {
      stride.d[i] = 1;
    }
    for (int i = 0; i < axes_size; i++) {
      stride.d[axes[i]] = strides[i];
    }

    nvinfer1::Dims size;
    size.nbDims = input_dims.nbDims;
    for (int i = 0; i < size.nbDims; i++) {
      size.d[i] = 1;
    }

    auto output_name = op_desc.Output("Out")[0];

    auto create_weights = [&](const std::vector<int>& data,
                              const std::string& type) -> int* {
      std::unique_ptr<framework::Tensor> tmp_tensor(new framework::Tensor());
      int data_size = data.size();
      tmp_tensor->Resize({data_size});
      auto* tmp_data = tmp_tensor->mutable_data<int>(platform::CPUPlace());
      for (int i = 0; i < data_size; i++) {
        tmp_data[i] = data[i];
      }

      engine_->SetWeights(output_name + "_add_slice_op_" + type,
                          std::move(tmp_tensor));
      return tmp_data;
    };

    std::vector<int> const_weight(input_dims.nbDims, 1);
    for (int i = 0; i < axes_size; i++) {
      const_weight[axes[i]] = strides[i];
    }

    int* weight_data = create_weights(const_weight, "size");

    TensorRTEngine::Weight weight{nvinfer1::DataType::kINT32,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(input_dims.nbDims)};

    int input_dim_size = input_dims.nbDims;
    auto const_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Constant, nvinfer1::Dims{1, {input_dim_size}}, weight.get());

    auto shape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input);

    auto size_layer = TRT_ENGINE_ADD_LAYER(
        engine_, ElementWise, *shape_layer->getOutput(0),
        *const_layer->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);

    auto* layer =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, start, size, stride);
    layer->setInput(2, *size_layer->getOutput(0));

    RreplenishLayerAndOutput(layer, "strided_slice", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(strided_slice, StridedSliceOpConverter);
