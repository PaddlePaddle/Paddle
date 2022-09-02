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

#define CHECK_SET_TYPE(want, fact)                                             \
   PADDLE_ENFORCE_EQ(                                                           \
       (want),                                                                  \
       (fact),                                                                  \
       platform::errors::InvalidArgument(                                       \
           "Errors occures in Paddle-TRT cast op, try to use C++ Api "          \
           "config.Exp_DisableTensorRtOPs({\"cast\"})\n; or Python Api "        \
           "config.exp_disable_tensorrt_ops([\"cast\"]) to forbid cat op into " \
           "Paddle-TRT."));


class LogicalNotOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(7000)
    VLOG(4) << "convert a fluid greater_equal op to tensorrt "
               "ElementWise layer";

    framework::OpDesc op_desc(op, nullptr);
    auto* x = engine_->GetITensor(op_desc.Input("X")[0]);
    std::cout << op_desc.Input("X")[0] << std::endl;

    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Unary, *x, nvinfer1::UnaryOperation::kNOT);

 layer->getOutput(0)->setType(nvinfer1::DataType::kBOOL);

CHECK_SET_TYPE(nvinfer1::DataType::kBOOL,
                        layer->getOutput(0)->getType());

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(
        layer, "logical_not", {output_name}, test_mode);
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(logical_not,
                          LogicalNotOpConverter);
