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

class MultiheadMatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid multihead_mamul op to a tensorrt network";
    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs
    auto* Q = engine_->GetITensor(op_desc.Input("Q").front());
    auto* K = engine_->GetITensor(op_desc.Input("K").front());
    auto* V = engine_->GetITensor(op_desc.Input("V").front());
    auto* BiasQ = engine_->GetITensor(op_desc.Input("BiasQ").front());
    auto* BiasK = engine_->GetITensor(op_desc.Input("BiasK").front());
    auto* BiasV = engine_->GetITensor(op_desc.Input("BiasV").front());
    auto* BiasQK = engine_->GetITensor(op_desc.Input("BiasQK").front());
    const bool transpose_q =
        op_desc.HasAttr("transpose_Q")
            ? boost::get<bool>(op_desc.GetAttr("transpose_Q"))
            : false;
    const bool transpose_k =
        op_desc.HasAttr("transpose_K")
            ? boost::get<bool>(op_desc.GetAttr("transpose_K"))
            : true;
    const bool transpose_v =
        op_desc.HasAttr("transpose_V")
            ? boost::get<bool>(op_desc.GetAttr("transpose_V"))
            : false;
    // Declare attributes
    const float alpha = boost::get<float>(op_desc.GetAttr("alpha"));
    const int head_number = boost::get<int>(op_desc.GetAttr("head_number"));
    nvinfer1::Dims q_shape = Q->getDimensions();
    int seq_len = q_shape.d[1];
    int size_per_head = q_shape.d[3];
    float* p_alpha;
    p_alpha[0] = alpha;

    TensorRTEngine::Weight scale{nvinfer1::DataType::kFLOAT,
                                 static_cast<void*>(p_alpha), 1};
    TensorRTEngine::Weight shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
    TensorRTEngine::Weight power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto* v_eltadd_layer = TRT_ENGINE_ADD_LAYER(
        engine_, ElementWise, *const_cast<nvinfer1::ITensor*>(V),
        *const_cast<nvinfer1::ITensor*>(BiasV),
        nvinfer1::ElementWiseOperation::kSUM);
    auto* q_eltadd_layer = TRT_ENGINE_ADD_LAYER(
        engine_, ElementWise, *const_cast<nvinfer1::ITensor*>(Q),
        *const_cast<nvinfer1::ITensor*>(BiasQ),
        nvinfer1::ElementWiseOperation::kSUM);
    auto* k_eltadd_layer = TRT_ENGINE_ADD_LAYER(
        engine_, ElementWise, *const_cast<nvinfer1::ITensor*>(K),
        *const_cast<nvinfer1::ITensor*>(BiasK),
        nvinfer1::ElementWiseOperation::kSUM);

    auto* v_transpose_reshape_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(v_eltadd_layer->getOutput(0)));
    auto* q_transpose_reshape_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(q_eltadd_layer->getOutput(0)));
    auto* k_transpose_reshape_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(k_eltadd_layer->getOutput(0)));

    nvinfer1::Dims4 head_reshape_dim(0, seq_len, head_number, size_per_head);

    v_transpose_reshape_layer->setReshapeDimensions(head_reshape_dim);
    v_transpose_reshape_layer->setSecondTranspose({0, 2, 1, 3});
    q_transpose_reshape_layer->setReshapeDimensions(head_reshape_dim);
    q_transpose_reshape_layer->setSecondTranspose({0, 2, 1, 3});
    k_transpose_reshape_layer->setReshapeDimensions(head_reshape_dim);
    k_transpose_reshape_layer->setSecondTranspose({0, 2, 1, 3});

    auto* k_scale_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Scale, *(k_transpose_reshape_layer->getOutput(0)),
        nvinfer1::ScaleMode::kUNIFORM, shift.get(), scale.get(), power.get());

    auto* qk_matmul_layer = TRT_ENGINE_ADD_LAYER(
        engine_, MatrixMultiply, *(k_scale_layer->getOutput(0)),
        transpose_k ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE,
        *(q_transpose_reshape_layer->getOutput(0)),
        transpose_q ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE);
    auto* qk_eltadd_layer = TRT_ENGINE_ADD_LAYER(
        engine_, ElementWise, *const_cast<nvinfer1::ITensor*>(BiasQK),
        *(qk_matmul_layer->getOutput(0)), nvinfer1::ElementWiseOperation::kSUM);
    auto* softmax_layer = TRT_ENGINE_ADD_LAYER(
        engine_, SoftMax, *(qk_eltadd_layer->getOutput(0)));
    auto* qkv_matmul_layer = TRT_ENGINE_ADD_LAYER(
        engine_, MatrixMultiply, *(softmax_layer->getOutput(0)),
        nvinfer1::MatrixOperation::kNONE,
        *(v_transpose_reshape_layer->getOutput(0)),
        transpose_v ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE);
    auto* qkv_transpose_reshape_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Shuffle, *(qkv_matmul_layer->getOutput(0)));

    nvinfer1::Dims4 qkv_reshape_dim(0, seq_len, head_number * size_per_head, 1);
    qkv_transpose_reshape_layer->setFirstTranspose({0, 2, 1, 3});
    qkv_transpose_reshape_layer->setReshapeDimensions(qkv_reshape_dim);

    auto output_name = op_desc.Output("Out").front();

    RreplenishLayerAndOutput(qkv_transpose_reshape_layer, "multihead_matmul",
                             {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(multihead_matmul, MultiheadMatMulOpConverter);
