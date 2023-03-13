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
#if IS_TRT_VERSION_GE(8200)
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
    // print padding
    std::cout << "[debug] paddings: ";
    for (auto& pad : paddings) {
      std::cout << pad << " ";
    }
    std::cout << std::endl;
    float value{0.F};
    if (op_desc.HasAttr("value")) {
      value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
    }

    std::string padding_mode = "constant";
    if (op_desc.HasAttr("mode")) {
      padding_mode = PADDLE_GET_CONST(std::string, op_desc.GetAttr("mode"));
    }
    std::cout << "[debug] padding_mode: " << padding_mode << std::endl;

    if (!engine_->with_dynamic_shape()) {
      nvinfer1::Dims reshape_dims = input->getDimensions();
      reshape_dims.nbDims = 5;
      reshape_dims.d[0] = 1;
      for (int i = 1; i < input->getDimensions().nbDims; i++) {
        reshape_dims.d[i] = input->getDimensions().d[i - 1];
      }
      auto reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
      reshape_layer->setReshapeDimensions(reshape_dims);
      input = reshape_layer->getOutput(0);
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

    for (int i = 0; i < input_dim; i += 2) {
      if (engine_->with_dynamic_shape()) {
        pre_pad_v[i + 2] = paddings[pad_size - 2 - i];
        post_pad_v[i + 2] = paddings[pad_size - 1 - i];
      } else {
        pre_pad_v[i + 1] = paddings[pad_size - 2 - i];
        post_pad_v[i + 1] = paddings[pad_size - 1 - i];
      }
    }
    std::cout << "[debug] pre_pad_v: ";
    for (auto& pad : pre_pad_v) {
      std::cout << pad << " ";
    }
    std::cout << std::endl;
    std::cout << "[debug] post_pad_v: ";
    for (auto& pad : post_pad_v) {
      std::cout << pad << " ";
    }
    std::cout << std::endl;
    std::cout << "[debug] break point 1" << std::endl;
    nvinfer1::ITensor* pre_pad = Add1DConstantLayer(pre_pad_v);
    nvinfer1::ITensor* post_pad = Add1DConstantLayer(post_pad_v);
    std::cout << "[debug] break point 2" << std::endl;
    std::vector<int> zeros_v(input_dim, 0);
    std::cout << "[debug] break point 3" << std::endl;
    auto const zeros = Add1DConstantLayer(zeros_v);
    // print dims of zeros and pre_pad
    std::cout << "[debug] zeros dims: ";
    for (int i = 0; i < zeros->getDimensions().nbDims; i++) {
      std::cout << zeros->getDimensions().d[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "[debug] pre_pad dims: ";
    for (int i = 0; i < pre_pad->getDimensions().nbDims; i++) {
      std::cout << pre_pad->getDimensions().d[i] << " ";
    }
    std::cout << std::endl;

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
    // print input shape
    std::cout << "[debug] input_shape: ";
    for (int i = 0; i < input_shape->getDimensions().nbDims; i++) {
      std::cout << input_shape->getDimensions().d[i] << " ";
    }
    std::cout << std::endl;
    // print total_padding shape
    std::cout << "[debug] total_padding: ";
    for (int i = 0; i < total_padding->getDimensions().nbDims; i++) {
      std::cout << total_padding->getDimensions().d[i] << " ";
    }
    std::cout << std::endl;

    size = TRT_ENGINE_ADD_LAYER(engine_,
                                ElementWise,
                                *input_shape,
                                *total_padding,
                                nvinfer1::ElementWiseOperation::kSUM)
               ->getOutput(0);
    std::cout << "[debug] break point1 here" << std::endl;
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
            auto* value_ptr = const_cast<float*>(&value);
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
    // print shape
    std::cout << "[debug] output shape: "
              << slice_layer->getOutput(0)->getDimensions().nbDims << "=>";

    for (int i = 0; i < slice_layer->getOutput(0)->getDimensions().nbDims;
         i++) {
      std::cout << slice_layer->getOutput(0)->getDimensions().d[i] << " ";
    }
    std::cout << std::endl;
    auto output_name = op_desc.Output("Out")[0];
    if (engine_->with_dynamic_shape()) {
      RreplenishLayerAndOutput(slice_layer, "pad3d", {output_name}, test_mode);
    } else {
      auto reshape_dims = slice_layer->getOutput(0)->getDimensions();
      for (int i = 0; i < reshape_dims.nbDims - 1; i++) {
        reshape_dims.d[i] = reshape_dims.d[i + 1];
      }
      reshape_dims.nbDims -= 1;
      auto reshape_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *slice_layer->getOutput(0));
      reshape_layer->setReshapeDimensions(reshape_dims);
      RreplenishLayerAndOutput(
          reshape_layer, "pad3d", {output_name}, test_mode);
    }

#else
    VLOG(3) << "pad3d is not supported when TensorRT < 8.2";
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(pad3d, Pad3dOpConverter);
