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

// Reorder the elements from istrides to ostrides, borrowed from TRT convert in
// tensorflow.
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/tensorrt/convert/convert_nodes.cc#L318
template <typename T>
void Reorder2(nvinfer1::DimsHW shape, const T* idata, nvinfer1::DimsHW istrides,
              T* odata, nvinfer1::DimsHW ostrides) {
  for (int h = 0; h < shape.h(); ++h) {
    for (int w = 0; w < shape.w(); ++w) {
      odata[h * ostrides.h() + w * ostrides.w()] =
          idata[h * istrides.h() + w * istrides.w()];
    }
  }
}
// indata c * k
// Reorder the data layout from CK to KC.
void ReorderCKtoKC(TensorRTEngine::Weight& iweights,  // NOLINT
                   TensorRTEngine::Weight* oweights) {
  int c = iweights.dims[0];
  int k = iweights.dims[1];
  oweights->dims.assign({k, c});
  nvinfer1::DimsHW istrides = {1, k};
  nvinfer1::DimsHW ostrides = {c, 1};
  Reorder2({k, c}, static_cast<float const*>(iweights.get().values), istrides,
           static_cast<float*>(const_cast<void*>(oweights->get().values)),
           ostrides);
}

/*
 * FC converter convert a MUL op in Fluid to a FC layer in TRT.
 */
class FcOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid fc op to tensorrt fc layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1);  // Y is a weight
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

    // Declare inputs
    auto* X = engine_->GetITensor(op_desc.Input("X").front());

    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input("Y").front());
    PADDLE_ENFORCE_NOT_NULL(Y_v);
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    // This may trigger a GPU->CPU copy, because TRT's weight can only be
    // assigned from CPU memory, that can't be avoided.
    float* weight_data = nullptr;
    bool enable_int8 = boost::get<bool>(op_desc.HasAttr("enable_int8"));
    if (enable_int8) {
#if IS_TRT_VERSION_GE(5000)
      float in_scale = boost::get<float>(op_desc.GetAttr("input_scale"));
      auto weight_scale = boost::get<float>(op_desc.GetAttr("weight_scale"));
      weight_data = engine_->GetWeightCPUData(op_desc.Input("Y").front(), Y_t,
                                              true, {weight_scale});
      engine_->SetTensorDynamicRange(X, in_scale);
#endif
    } else {
      weight_data =
          engine_->GetWeightCPUData(op_desc.Input("Y").front(), Y_t, false);
    }

    PADDLE_ENFORCE_EQ(Y_t->dims().size(), 2UL);  // a matrix
    size_t n_output = Y_t->dims()[1];

    std::unique_ptr<framework::Tensor> tmp(new framework::LoDTensor());
    tmp->Resize(Y_t->dims());

    memcpy(tmp->mutable_data<float>(platform::CPUPlace()), weight_data,
           Y_t->dims()[0] * Y_t->dims()[1] * sizeof(float));
    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(Y_t->numel())};
    TensorRTEngine::Weight tmp_weight(nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(tmp->data<float>()),
                                      static_cast<size_t>(Y_t->numel()));
    weight.dims.assign({Y_t->dims()[0], Y_t->dims()[1]});
    tmp_weight.dims = weight.dims;

    // The data layout of TRT FC layer's weight is different from fluid's FC,
    // need to reorder the elements.
    ReorderCKtoKC(weight, &tmp_weight);

    // Currently, the framework can only handle one fluid op -> one TRT layer,
    // but fc fuses `mul` and `bias` (2 fluid ops), so here is a trick, just
    // handle `mul`, leave `add` as another layer.
    // DEBUG
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected,
                                       *const_cast<nvinfer1::ITensor*>(X),
                                       n_output, tmp_weight.get(), bias.get());

    engine_->weight_map[op_desc.Input("Y").front()] = std::move(tmp);
    auto output_name = op_desc.Output("Out").front();

    RreplenishLayerAndOutput(layer, "fc", {output_name}, test_mode);
    if (enable_int8) {
#if IS_TRT_VERSION_GE(5000)
      float out_scale = boost::get<float>(op_desc.GetAttr("output_scale"));
      engine_->SetTensorDynamicRange(layer->getOutput(0), out_scale);
#endif
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);
