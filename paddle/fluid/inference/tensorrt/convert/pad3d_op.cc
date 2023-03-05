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
 * Pad3dOp.
 */
class Pad3dOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid transpose op to tensorrt tranpose layer";

    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    std::vector<int> paddings;
    if (op_desc.HasInput("Paddings")) {
      auto* paddings_v = scope.FindVar(op_desc.Input("Paddings")[0]);
      auto* padding_t = paddings_v->GetMutable<phi::DenseTensor>();
      phi::DenseTensor paddings_tensor;
      paddings_tensor.Resize(padding_t->dims());
      platform::CPUPlace cpu_place;
      paddle::framework::TensorCopySync(
          (*padding_t), cpu_place, &paddings_tensor);
      auto* paddings_data =
          paddings_tensor.mutable_data<int>(platform::CPUPlace());
      paddings = std::vector<int>(paddings_data,
                                  paddings_data + paddings_tensor.numel());

    } else {
      paddings =
          PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
    }

    float value{0.F};
    if (op_desc.HasAttr("value")) {
      value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
    }

    std::string padding_mode;
    if (op_desc.HasAttr("mode")) {
      padding_mode = PADDLE_GET_CONST(std::string, op_desc.GetAttr("mode"));
    } else {
      padding_mode = "constant";
    }

    const int inputDim = input->getDimensions().nbDims;
    const int pad_size = paddings.size();
    PADDLE_ENFORCE_EQ(inputDim * 2 - 4,
                      pad_size,
                      phi::errors::InvalidArgument(
                          "Expected paddings size is %d, but received %d.",
                          inputDim * 2 - 4,
                          pad_size));

    // convert paddle pad to tensorrt pad
    std::vector<int> pre_pad_v(inputDim, 0);
    std::vector<int> post_pad_v(inputDim, 0);

    for (int i = 0; i < inputDim; i += 2) {
      pre_pad_v[i + 2] = paddings[pad_size - 2 - i];
      post_pad_v[i + 2] = paddings[pad_size - 1 - i];
    }

    nvinfer1::ITensor* pre_pad = vectorToTensor<int>(pre_pad_v);
    nvinfer1::ITensor* post_pad = vectorToTensor<int>(post_pad_v);

    std::vector<int> zeros_v(inputDim, 0);
    auto const zeros = vectorToTensor<int>(zeros_v);

    nvinfer1::ITensor* start{};
    nvinfer1::ITensor* size{};
    // elementwise add zeros and pre_pad
    start = TRT_ENGINE_ADD_LAYER(engine_,
                                 ElementWise,
                                 *zeros,
                                 *pre_pad,
                                 nvinfer1::ElementWiseOperation::kSUB)
                ->getOutput(0);

    auto const total_padding =
        TRT_ENGINE_ADD_LAYER(engine_,
                             ElementWise,
                             *pre_pad,
                             *post_pad,
                             nvinfer1::ElementWiseOperation::kSUM)
            ->getOutput(0);

    std::vector<int> input_shape_v(inputDim, 0);
    for (int i = 0; i < inputDim; i++) {
      input_shape_v[i] = input->getDimensions().d[i];
    }
    auto const input_shape = vectorToTensor<int>(input_shape_v);

    size = TRT_ENGINE_ADD_LAYER(engine_,
                                ElementWise,
                                *input_shape,
                                *total_padding,
                                nvinfer1::ElementWiseOperation::kSUM)
               ->getOutput(0);

    // add slice layer
    nvinfer1::Dims stride;
    stride.nbDims = inputDim;
    std::fill_n(stride.d, inputDim, 1);
    auto const& dummy = stride;
    auto* slice_layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Slice,
                             *const_cast<nvinfer1::ITensor*>(input),
                             dummy,
                             dummy,
                             stride);
    slice_layer->setInput(1, *start);
    slice_layer->setInput(2, *size);
    if (padding_mode == "constant") {
      slice_layer->setMode(nvinfer1::SliceMode::kFILL);
      if (value != 0.F) {
        nvinfer1::ITensor* fill_value = nullptr;
        switch (input->getType()) {
          case nvinfer1::DataType::kFLOAT:
          case nvinfer1::DataType::kHALF:
          case nvinfer1::DataType::kINT8: {
            float* value_ptr = const_cast<float*>(&value);
            nvinfer1::Weights value_wt{nvinfer1::DataType::kFLOAT,
                                       static_cast<void*>(value_ptr),
                                       static_cast<int32_t>(1)};
            nvinfer1::Dims dims;
            dims.nbDims = 0;
            fill_value = TRT_ENGINE_ADD_LAYER(engine_, Constant, dims, value_wt)
                             ->getOutput(0);
          }
          default: {
            int* value_ptr = const_cast<int*>(reinterpret_cast<int*>(&value));
            nvinfer1::Weights value_wt{nvinfer1::DataType::kINT32,
                                       static_cast<void*>(value_ptr),
                                       static_cast<int32_t>(1)};
            nvinfer1::Dims dims;
            dims.nbDims = 0;
            fill_value = TRT_ENGINE_ADD_LAYER(engine_, Constant, dims, value_wt)
                             ->getOutput(0);
          }
        }
        slice_layer->setInput(4, *fill_value);
      }
    } else if (padding_mode == "reflect") {
      slice_layer->setMode(nvinfer1::SliceMode::kREFLECT);
    } else if (padding_mode == "replicate") {
      slice_layer->setMode(nvinfer1::SliceMode::kCLAMP);
    } else {
      PADDLE_THROW("Unsupported mode: %s", padding_mode);
    }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(slice_layer, "pad3d", {output_name}, test_mode);
  }

 private:
  template <typename T>
  nvinfer1::ITensor* vectorToTensor(std::vector<T> v) {
    int* v_data = const_cast<T*>(static_cast<const T*>(v.data()));

    nvinfer1::Weights v_wt{nvinfer1::DataType::kINT32,
                           static_cast<void*>(v_data),
                           static_cast<int32_t>(v.size())};

    nvinfer1::Dims v_dim;
    v_dim.nbDims = 1;
    v_dim.d[0] = static_cast<int>(v.size());

    return TRT_ENGINE_ADD_LAYER(engine_, Constant, v_dim, v_wt)->getOutput(0);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(pad3d, Pad3dOpConverter);
