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
    auto* input = engine_->GetITensor(op_desc.Input("X").front());

    nvinfer1::ITensor* paddings;

    if (engine_->with_dynamic_shape() && op_desc.HasInput("Paddings") &&
        op_desc.Input("Paddings").size() >= 1) {
      paddings = engine_->GetITensor(op_desc.Input("Paddings").front());
      //      auto* paddings_v =
      //      scope.FindVar(op_desc.Input("Paddings").front());
      //      PADDLE_ENFORCE_NOT_NULL(
      //          paddings_v,
      //          platform::errors::NotFound(
      //              "Variable of Paddings of pad3d TRT converter is not
      //              found."));
      //      auto* padding_t = paddings_v->GetMutable<phi::DenseTensor>();
      //      auto* p = padding_t;
      //      auto* padding_d = p->data<int>();
      //      for (int i = 0; i < padding_t->numel(); i++) {
      //        paddings.push_back(padding_d[i]);
      //      }
    } else {
      std::vector<int> paddings_v =
          PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
      paddings = Add1DConstantLayer(paddings_v);
    }

    float value{0.F};
    if (op_desc.HasAttr("value")) {
      value = PADDLE_GET_CONST(float, op_desc.GetAttr("value"));
    }

    std::string padding_mode = "constant";
    if (op_desc.HasAttr("mode")) {
      padding_mode = PADDLE_GET_CONST(std::string, op_desc.GetAttr("mode"));
    }

    const int inputDim = input->getDimensions().nbDims;
    const int pad_size = paddings->getDimensions().d[0];
    PADDLE_ENFORCE_EQ(inputDim * 2 - 4,
                      pad_size,
                      phi::errors::InvalidArgument(
                          "Expected paddings size is %d, but received %d.",
                          inputDim * 2 - 4,
                          pad_size));

    // convert paddle pad to tensorrt pad
    //    auto transpose_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle,
    //    *paddings); nvinfer1::Permutation perm{4, 2, 0, 5, 3, 1};
    //    transpose_layer->setFirstTranspose(perm);
    //    paddings = transpose_layer->getOutput(0);
    // split paddings to pre_pad and post_pad
    auto* pre_zeros = Add1DConstantLayer(std::vector<int>(2, 0));
    auto start_slice1 = nvinfer1::Dims{1, { 0 }};
    auto start_slice2 = nvinfer1::Dims{1, { 3 }};
    auto size_slice = nvinfer1::Dims{1, { 3 }};
    auto stride_slice = nvinfer1::Dims{1, { 1 }};
    auto* pre_pad =
        TRT_ENGINE_ADD_LAYER(
            engine_, Slice, *paddings, start_slice1, size_slice, stride_slice)
            ->getOutput(0);
    pre_pad = Concat(std::vector<nvinfer1::ITensor*>{pre_zeros, pre_pad});
    auto* post_pad =
        TRT_ENGINE_ADD_LAYER(
            engine_, Slice, *paddings, start_slice2, size_slice, stride_slice)
            ->getOutput(0);
    post_pad = Concat(std::vector<nvinfer1::ITensor*>{pre_zeros, post_pad});
    std::cout << "pre_pad: " << pre_pad->getDimensions().d[0] << std::endl;
    std::cout << "post_pad: " << post_pad->getDimensions().d[0] << std::endl;
    std::vector<int> zeros_v(inputDim, 0);
    auto const zeros = Add1DConstantLayer(zeros_v);
    // elementwise add zeros and pre_pad
    nvinfer1::ITensor* start =
        TRT_ENGINE_ADD_LAYER(engine_,
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
    auto const input_shape = Add1DConstantLayer(input_shape_v);

    nvinfer1::ITensor* size =
        TRT_ENGINE_ADD_LAYER(engine_,
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

#else
    VLOG(3) << "pad3d is not supported when TensorRT < 8.2";
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(pad3d, Pad3dOpConverter);
