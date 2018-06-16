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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SoftmaxOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    op_desc.SetAttr("is_test", true);
    op_desc.SetAttr("data_format", "NHWC");
    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

    auto x_name = op_desc.Input("X")[0];
    auto out_name = op_desc.Output("Out")[0];

    // Declare inputs.
    auto* x = engine_->GetITensor(x_name);
    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, SoftMax,
                                       *const_cast<nvinfer1::ITensor*>(x));
    engine_->SetITensor(out_name, layer->getOutput(0));
    if (test_mode) {
      engine_->DeclareOutput(out_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(softmax, SoftmaxOpConverter);
USE_OP(softmax);
