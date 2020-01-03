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
    VLOG(3) << "convert a fluid multihead_mamul op to a corresponding tensorrt "
               "network structure";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* Q = engine_->GetITensor(op_desc.Input("Q").front());
    auto* K = engine_->GetITensor(op_desc.Input("K").front());
    auto* V = engine_->GetITensor(op_desc.Input("V").front());
    auto* BiasQ = scope.FindVar(op_desc.Input("BiasQ").front());
    auto* BiasK = scope.FindVar(op_desc.Input("BiasK").front());
    auto* BiasV = scope.FindVar(op_desc.Input("BiasV").front());
    auto* BiasQK = engine_->GetITensor(op_desc.Input("BiasQK").front());
    PADDLE_ENFORCE_EQ(op_desc.Input("Q").size(), 1,
                      platform::errors::InvalidArgument(
                          "size of input Q of multihead_matmul should be 1"));
    PADDLE_ENFORCE_EQ(op_desc.Input("K").size(), 1,
                      platform::errors::InvalidArgument(
                          "size of input K of multihead_matmul should be 1"));
    PADDLE_ENFORCE_EQ(op_desc.Input("V").size(), 1,
                      platform::errors::InvalidArgument(
                          "size of input V of multihead_matmul should be 1"));
    PADDLE_ENFORCE_EQ(
        op_desc.Input("BiasQK").size(), 1,
        platform::errors::InvalidArgument(
            "size of input BiasQK of multihead_matmul should be 1"));
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1,
                      platform::errors::InvalidArgument(
                          "size of output of multihead_matmul should be 1"));
    PADDLE_ENFORCE_NOT_NULL(
        BiasQ, platform::errors::InvalidArgument(
                   "param BiasQ of multihead_matmul should not be null"));
    PADDLE_ENFORCE_NOT_NULL(
        BiasK, platform::errors::InvalidArgument(
                   "param BiasK of multihead_matmul should not be null"));
    PADDLE_ENFORCE_NOT_NULL(
        BiasV, platform::errors::InvalidArgument(
                   "param BiasV of multihead_matmul should not be null"));
    PADDLE_ENFORCE_EQ(
        BiasQK->getDimensions().nbDims, 3,
        platform::errors::InvalidArgument(
            "dims size of input BiasQK of multihead_matmul should be 3"));
    PADDLE_ENFORCE_EQ(
        op_desc.HasAttr("alpha"), true,
        platform::errors::PreconditionNotMet(
            "attribute alpha of multihead_matmul should not be empty"));
    PADDLE_ENFORCE_EQ(
        op_desc.HasAttr("head_number"), true,
        platform::errors::PreconditionNotMet(
            "attribute head_number of multihead_matmul should not be empty"));

    // Declare attributes
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
    const float alpha = boost::get<float>(op_desc.GetAttr("alpha"));
    const int head_number = boost::get<int>(op_desc.GetAttr("head_number"));

    nvinfer1::Dims q_shape = Q->getDimensions();
    int seq_len = q_shape.d[0];
    int size_per_head = q_shape.d[1] / head_number;
    std::string alpha_name = op_desc.Output("Out")[0] + "_alpha";
    framework::DDim alpha_dim = framework::make_ddim({1});
    std::unique_ptr<framework::LoDTensor> alpha_t(new framework::LoDTensor());
    alpha_t->Resize(alpha_dim);
    float* alpha_data = alpha_t->mutable_data<float>(platform::CPUPlace());
    alpha_data[0] = alpha;

    TensorRTEngine::Weight scale{nvinfer1::DataType::kFLOAT,
                                 static_cast<void*>(alpha_data), 1};
    TensorRTEngine::Weight shift{nvinfer1::DataType::kFLOAT, nullptr, 0};
    TensorRTEngine::Weight power{nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto* bias_q_t = BiasQ->GetMutable<framework::LoDTensor>();
    auto* bias_k_t = BiasK->GetMutable<framework::LoDTensor>();
    auto* bias_v_t = BiasV->GetMutable<framework::LoDTensor>();
    float* bias_q_cpu_data = engine_->GetWeightCPUData(
        op_desc.Input("BiasQ").front(), bias_q_t, false);
    float* bias_k_cpu_data = engine_->GetWeightCPUData(
        op_desc.Input("BiasK").front(), bias_k_t, false);
    float* bias_v_cpu_data = engine_->GetWeightCPUData(
        op_desc.Input("BiasV").front(), bias_v_t, false);
    std::unique_ptr<framework::LoDTensor> bias_q_tensor(
        new framework::LoDTensor());
    std::unique_ptr<framework::LoDTensor> bias_k_tensor(
        new framework::LoDTensor());
    std::unique_ptr<framework::LoDTensor> bias_v_tensor(
        new framework::LoDTensor());
    bias_q_tensor->Resize(bias_q_t->dims());
    bias_k_tensor->Resize(bias_k_t->dims());
    bias_v_tensor->Resize(bias_v_t->dims());
    platform::CPUPlace cpu_place;
    TensorCopySync((*bias_q_t), cpu_place, bias_q_tensor.get());
    TensorCopySync((*bias_k_t), cpu_place, bias_k_tensor.get());
    TensorCopySync((*bias_v_t), cpu_place, bias_v_tensor.get());

    TensorRTEngine::Weight scale_weights_q{nvinfer1::DataType::kFLOAT, nullptr,
                                           0};
    TensorRTEngine::Weight shift_weights_q{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(bias_q_cpu_data),
        bias_q_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight power_weights_q{nvinfer1::DataType::kFLOAT, nullptr,
                                           0};
    TensorRTEngine::Weight scale_weights_k{nvinfer1::DataType::kFLOAT, nullptr,
                                           0};
    TensorRTEngine::Weight shift_weights_k{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(bias_k_cpu_data),
        bias_k_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight power_weights_k{nvinfer1::DataType::kFLOAT, nullptr,
                                           0};
    TensorRTEngine::Weight scale_weights_v{nvinfer1::DataType::kFLOAT, nullptr,
                                           0};
    TensorRTEngine::Weight shift_weights_v{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(bias_v_cpu_data),
        bias_v_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight power_weights_v{nvinfer1::DataType::kFLOAT, nullptr,
                                           0};

    auto* q_eltadd_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Scale, *Q, nvinfer1::ScaleMode::kCHANNEL,
        shift_weights_q.get(), scale_weights_q.get(), power_weights_q.get());
    auto* k_eltadd_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Scale, *K, nvinfer1::ScaleMode::kCHANNEL,
        shift_weights_k.get(), scale_weights_k.get(), power_weights_k.get());
    auto* v_eltadd_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Scale, *V, nvinfer1::ScaleMode::kCHANNEL,
        shift_weights_v.get(), scale_weights_v.get(), power_weights_v.get());
    auto* v_transpose_reshape_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(v_eltadd_layer->getOutput(0)));
    auto* q_transpose_reshape_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(q_eltadd_layer->getOutput(0)));
    auto* k_transpose_reshape_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(k_eltadd_layer->getOutput(0)));

    nvinfer1::Dims3 head_reshape_dim(seq_len, head_number, size_per_head);
    v_transpose_reshape_layer->setReshapeDimensions(head_reshape_dim);
    v_transpose_reshape_layer->setSecondTranspose({1, 0, 2});
    q_transpose_reshape_layer->setReshapeDimensions(head_reshape_dim);
    q_transpose_reshape_layer->setSecondTranspose({1, 0, 2});
    k_transpose_reshape_layer->setReshapeDimensions(head_reshape_dim);
    k_transpose_reshape_layer->setSecondTranspose({1, 0, 2});

    auto* q_scale_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Scale, *(q_transpose_reshape_layer->getOutput(0)),
        nvinfer1::ScaleMode::kUNIFORM, shift.get(), scale.get(), power.get());
    auto* qk_matmul_layer = TRT_ENGINE_ADD_LAYER(
        engine_, MatrixMultiply, *(q_scale_layer->getOutput(0)), transpose_q,
        *(k_transpose_reshape_layer->getOutput(0)), transpose_k);
    auto* qk_eltadd_layer = TRT_ENGINE_ADD_LAYER(
        engine_, ElementWise, *BiasQK, *(qk_matmul_layer->getOutput(0)),
        nvinfer1::ElementWiseOperation::kSUM);
    auto* softmax_layer = TRT_ENGINE_ADD_LAYER(
        engine_, SoftMax, *(qk_eltadd_layer->getOutput(0)));
    softmax_layer->setAxes(4);
    auto* qkv_matmul_layer = TRT_ENGINE_ADD_LAYER(
        engine_, MatrixMultiply, *(softmax_layer->getOutput(0)), false,
        *(v_transpose_reshape_layer->getOutput(0)), transpose_v);
    auto* qkv_transpose_reshape_layer = TRT_ENGINE_ADD_LAYER(
        engine_, Shuffle, *(qkv_matmul_layer->getOutput(0)));
    nvinfer1::Dims2 qkv_reshape_dim(seq_len, head_number * size_per_head);
    qkv_transpose_reshape_layer->setFirstTranspose({1, 0, 2});
    qkv_transpose_reshape_layer->setReshapeDimensions(qkv_reshape_dim);

    engine_->SetWeights(alpha_name, std::move(alpha_t));
    engine_->SetWeights(op_desc.Input("BiasQ").front(),
                        std::move(bias_q_tensor));
    engine_->SetWeights(op_desc.Input("BiasK").front(),
                        std::move(bias_k_tensor));
    engine_->SetWeights(op_desc.Input("BiasV").front(),
                        std::move(bias_v_tensor));

    auto output_name = op_desc.Output("Out").front();
    RreplenishLayerAndOutput(qkv_transpose_reshape_layer, "multihead_matmul",
                             {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(multihead_matmul, MultiheadMatMulOpConverter);
