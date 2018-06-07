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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/platform/place.h"

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
          idata[h * ostrides.h() + w * ostrides.w()];
    }
  }
}

// Reorder the data layout from CK to KC.
void ReorderCKtoKC(TensorRTEngine::Weight& iweights,
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
    VLOG(4) << "convert a fluid fc op to tensorrt fc layer without bias";

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
    auto* weight_data = Y_t->mutable_data<float>(platform::CPUPlace());
    PADDLE_ENFORCE_EQ(Y_t->dims().size(), 2UL);  // a matrix
    size_t n_output = Y_t->dims()[1];

    framework::LoDTensor tmp;
    tmp.Resize(Y_t->dims());
    memcpy(tmp.mutable_data<float>(platform::CPUPlace()), Y_t->data<float>(),
           Y_t->dims()[0] * Y_t->dims()[1]);

    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  Y_t->memory_size() / sizeof(float)};
    TensorRTEngine::Weight tmp_weight(nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(tmp.data<float>()),
                                      Y_t->memory_size() / sizeof(float));
    weight.dims.assign({Y_t->dims()[0], Y_t->dims()[1]});
    tmp_weight.dims = weight.dims;

    // The data layout of TRT FC layer's weight is different from fluid's FC,
    // need to reorder the elements.
    ReorderCKtoKC(tmp_weight, &weight);

    // Currently, the framework can only handle one fluid op -> one TRT layer,
    // but fc fuses `mul` and `bias` (2 fluid ops), so here is a trick, just
    // handle `mul`, leave `add` as another layer.
    // DEBUG
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT, nullptr, 0};

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected,
                                       *const_cast<nvinfer1::ITensor*>(X),
                                       n_output, weight.get(), bias.get());

    auto output_name = op_desc.Output("Out").front();
    engine_->SetITensor(output_name, layer->getOutput(0));
    if (test_mode) {
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);
USE_OP(mul);
