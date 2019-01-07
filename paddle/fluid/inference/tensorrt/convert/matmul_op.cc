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
 * MatMulOp, IMatrixMultiplyLayer in TRT.
 */
class MatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid mul op to tensorrt mul layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    nvinfer1::ILayer* layer = nullptr;
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);
    bool transpose_x = boost::get<bool>(op_desc.GetAttr("transpose_X"));
    bool transpose_y = boost::get<bool>(op_desc.GetAttr("transpose_Y"));
    float alpha = boost::get<float>(op_desc.GetAttr("alpha"));

    auto* matmul_layer = TRT_ENGINE_ADD_LAYER(
        engine_, MatrixMultiply, *const_cast<nvinfer1::ITensor*>(input1), false,
        *const_cast<nvinfer1::ITensor*>(input2), false);

    if (transpose_x == true) {
      matmul_layer->setTranspose(0, true);
    }
    if (transpose_y == true) {
      matmul_layer->setTranspose(1, true);
    }

    layer = matmul_layer;
    if (alpha != 1.0) {
      auto matmul_output = layer->getOutput(0);
      auto matmul_dims = matmul_output->getDimensions();
      bool need_to_expand_dims = (matmul_dims.nbDims == 2);
      if (need_to_expand_dims) {
        nvinfer1::DimsCHW reshape_dims(1, matmul_dims.d[0], matmul_dims.d[1]);
        auto* reshape_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
        reshape_layer->setReshapeDimensions(reshape_dims);
        layer = reshape_layer;
      }

      platform::CPUPlace place;
      std::unique_ptr<framework::LoDTensor> alpha_tensor(
          new framework::LoDTensor());
      alpha_tensor->Resize(framework::make_ddim({1}));
      float* alpha_data = alpha_tensor->mutable_data<float>(place);
      alpha_data[0] = alpha;

      TensorRTEngine::Weight scale{nvinfer1::DataType::kFLOAT,
                                   static_cast<void*>(alpha_data), 1};
      TensorRTEngine::Weight shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
      TensorRTEngine::Weight power{nvinfer1::DataType::kFLOAT, nullptr, 0};
      layer = TRT_ENGINE_ADD_LAYER(engine_, Scale, *layer->getOutput(0),
                                   nvinfer1::ScaleMode::kUNIFORM, shift.get(),
                                   scale.get(), power.get());

      std::string alpha_name = "matmul_" + op_desc.Output("Out")[0];
      engine_->weight_map[alpha_name] = std::move(alpha_tensor);
      if (need_to_expand_dims) {
        auto* reshape_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
        reshape_layer->setReshapeDimensions(matmul_dims);
        layer = reshape_layer;
      }
    }

    auto output_name = op_desc.Output("Out")[0];
    engine_->SetITensor(output_name, layer->getOutput(0));
    if (test_mode) {  // the test framework can not determine which is the
                      // output, so place the declaration inside.
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(matmul, MatMulOpConverter);
