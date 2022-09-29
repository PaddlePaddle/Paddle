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
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid StridedSlice op to tensorrt Slice layer";

    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    nvinfer1::Dims input_dims = input->getDimensions();
    auto output_name = op_desc.Output("Out")[0];
    std::vector<int> axes =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    std::vector<int> starts =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("starts"));
    std::vector<int> ends =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("ends"));
    std::vector<int> strides =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
    int axes_size = axes.size();
    nvinfer1::Dims start;
    nvinfer1::Dims stride;
    nvinfer1::Dims size;
    start.nbDims = input_dims.nbDims;
    stride.nbDims = input_dims.nbDims;
    size.nbDims = input_dims.nbDims;
    for (int i = 0; i < input_dims.nbDims; i++) {
      start.d[i] = 0;
      stride.d[i] = 1;
      size.d[i] = input_dims.d[i];
    }

    if (!engine_->with_dynamic_shape()) {
      for (int i = 0; i < axes_size; i++) {
        start.d[axes[i] - 1] = starts[i];
      }
      for (int i = 0; i < axes_size; i++) {
        stride.d[axes[i] - 1] = strides[i];
      }
      for (int i = 0; i < axes_size; ++i) {
        int dim = size.d[axes[i] - 1];
        if (dim > 0) {
          int start = starts[i] < 0 ? (starts[i] + dim) : starts[i];
          int end = ends[i] < 0 ? (ends[i] + dim) : ends[i];
          int stride = std::abs(strides[i]);
          start = std::max(start, 0);
          end = std::max(end, 0);
          end = std::min(end, dim);
          size.d[axes[i] - 1] = (std::abs(end - start) + stride - 1) / stride;
        }
      }
      auto* layer =
          TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, start, size, stride);
      RreplenishLayerAndOutput(
          layer, "strided_slice", {output_name}, test_mode);
    } else {
      for (int i = 0; i < axes_size; i++) {
        start.d[axes[i]] = starts[i];
      }
      for (int i = 0; i < axes_size; i++) {
        stride.d[axes[i]] = strides[i];
      }
      for (int i = 0; i < axes_size; ++i) {
        int dim = size.d[axes[i]];
        if (dim > 0) {
          int start = starts[i] < 0 ? (starts[i] + dim) : starts[i];
          int end = ends[i] < 0 ? (ends[i] + dim) : ends[i];
          int stride = std::abs(strides[i]);
          start = std::max(start, 0);
          end = std::max(end, 0);
          end = std::min(end, dim);
          size.d[axes[i]] = (std::abs(end - start) + stride - 1) / stride;
        }
      }

      auto create_weights = [&](const std::vector<int>& data,
                                const std::string& type) -> int* {
        std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
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

      std::vector<int> const_weight(input_dims.nbDims, 0);
      for (int i = 0; i < axes_size; i++) {
        int dim = input_dims.d[axes[i]];
        int start = starts[i] < 0 ? (starts[i] + dim) : starts[i];
        int end = ends[i] < 0 ? (ends[i] + dim) : ends[i];
        int stride = std::abs(strides[i]);
        start = std::max(start, 0);
        end = std::max(end, 0);
        end = std::min(end, dim);
        const_weight[axes[i]] =
            dim - ((std::abs(end - start) + stride - 1) / stride);
      }

      int* weight_data = create_weights(const_weight, "size");

      TensorRTEngine::Weight weight{nvinfer1::DataType::kINT32,
                                    static_cast<void*>(weight_data),
                                    static_cast<size_t>(input_dims.nbDims)};

      int input_dim_size = input_dims.nbDims;
      nvinfer1::Dims input_shape;
      input_shape.nbDims = 1;
      input_shape.d[0] = input_dim_size;

      auto const_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, input_shape, weight.get());

      auto shape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input);
      // slice layer
      auto* layer =
          TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, start, size, stride);
      // elementwise layer for get size tensor
      auto size_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *shape_layer->getOutput(0),
                               *const_layer->getOutput(0),
                               nvinfer1::ElementWiseOperation::kSUB);
      layer->setInput(2, *size_layer->getOutput(0));
      RreplenishLayerAndOutput(
          layer, "strided_slice", {output_name}, test_mode);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(strided_slice, StridedSliceOpConverter);
