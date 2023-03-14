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
#if IS_TRT_VERSION_GE(8200)
    VLOG(3) << "convert a transpose op to tensorrt tranpose layer";

    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    std::vector<int> paddings;
    if (op_desc.HasInput("Paddings")) {
      // convert nvinfer1::ITensor to std::vector<int>
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

    std::string padding_mode = "constant";
    if (op_desc.HasAttr("mode")) {
      padding_mode = PADDLE_GET_CONST(std::string, op_desc.GetAttr("mode"));
    }

    const int input_dim = input->getDimensions().nbDims;
    const int pad_size = paddings.size();
    PADDLE_ENFORCE_EQ(input_dim * 2 - 4,
                      pad_size,
                      phi::errors::InvalidArgument(
                          "Expected paddings size is %d, but received %d.",
                          input_dim * 2 - 4,
                          pad_size));
    // convert paddle pad to tensorrt pad
    std::vector<int> pre_pad_v(input_dim, 0);
    std::vector<int> post_pad_v(input_dim, 0);

    for (int i = 0; i < input_dim - 2; i++) {
      pre_pad_v[i + 2] = paddings[pad_size - 2 - i * 2];
      post_pad_v[i + 2] = paddings[pad_size - 1 - i * 2];
    }

    nvinfer1::ITensor* pre_pad = Add1DConstantLayer(pre_pad_v);
    nvinfer1::ITensor* post_pad = Add1DConstantLayer(post_pad_v);
    std::vector<int> zeros_v(input_dim, 0);
    auto const zeros = Add1DConstantLayer(zeros_v);

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

    auto* input_shape = Shape(input);
    size = TRT_ENGINE_ADD_LAYER(engine_,
                                ElementWise,
                                *input_shape,
                                *total_padding,
                                nvinfer1::ElementWiseOperation::kSUM)
               ->getOutput(0);
    // add slice layer
    nvinfer1::Dims stride;
    stride.nbDims = input_dim;
    std::fill_n(stride.d, input_dim, 1);
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
#if IS_TRT_VERSION_GE(8500)
      slice_layer->setMode(nvinfer1::SampleMode::kFILL);
#else
      slice_layer->setMode(nvinfer1::SliceMode::kFILL);
#endif
      if (value != 0.F) {
        nvinfer1::ITensor* fill_value = nullptr;
        switch (input->getType()) {
          case nvinfer1::DataType::kFLOAT:
          case nvinfer1::DataType::kHALF:
          case nvinfer1::DataType::kINT8: {
            fill_value = Add1DConstantLayer(value);
            break;
          }
          default: {
            int value_int = static_cast<int>(value);
            fill_value = Add1DConstantLayer(value_int);
            break;
          }
        }
        slice_layer->setInput(4, *fill_value);
      }
    } else if (padding_mode == "reflect") {
#if IS_TRT_VERSION_GE(8500)
      slice_layer->setMode(nvinfer1::SampleMode::kREFLECT);
#else
      slice_layer->setMode(nvinfer1::SliceMode::kREFLECT);
#endif
    } else if (padding_mode == "replicate") {
#if IS_TRT_VERSION_GE(8500)
      slice_layer->setMode(nvinfer1::SampleMode::kCLAMP);
#else
      slice_layer->setMode(nvinfer1::SliceMode::kCLAMP);
#endif
    } else {
      PADDLE_THROW("Unsupported mode: %s", padding_mode);
    }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(slice_layer, "pad3d", {output_name}, test_mode);

#else
    VLOG(3) << "pad3d is not supported when TensorRT < 8.2";
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(pad3d, Pad3dOpConverter);
