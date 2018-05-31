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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * FC converter convert a MUL op in Fluid to a FC layer in TRT.
 */
class FcOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope) override {
    VLOG(4) << "convert a fluid fc op to tensorrt fc layer without bias";

    framework::OpDesc op_desc(op, nullptr, nullptr);
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1);     // Y is a weight
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);  // Y is a weight

    // Declare inputs
    auto* X = engine_->GetITensor(op_desc.Input("X").front());

    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input("Y").front());
    PADDLE_ENFORCE_NOT_NULL(Y_v);
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    // This may trigger a CPU->GPU copy.
    // TODO(Superjomn) use some smarter mutable_data.
    auto* weight_data = Y_t->mutable_data<float>(platform::CPUPlace());
    PADDLE_ENFORCE_EQ(Y_t->dims().size(), 2UL);  // a matrix
    size_t n_output = Y_t->dims()[1];

    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  Y_t->memory_size() / sizeof(float)};

    // Currently, the framework can only handle one fluid op -> one TRT layer,
    // but fc fuses `mul` and `bias` (2 fluid ops), so here is a trick, just
    // handle `mul`, leave `add` as another layer.
    // DEBUG
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT, nullptr,
                                0};

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, FullyConnected,
                                       *const_cast<nvinfer1::ITensor*>(X),
                                       n_output, weight.get(), bias.get());

    auto output_name = op_desc.Output("Out").front();
    engine_->DeclareOutput(layer, 0, output_name);
    auto* output = engine_->GetITensor(output_name);
    LOG(INFO) << "output dim";
    for (int i = 0; i < output->getDimensions().nbDims; i++) {
      LOG(INFO) << output->getDimensions().d[i];
    }
  }
};

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(mul);
