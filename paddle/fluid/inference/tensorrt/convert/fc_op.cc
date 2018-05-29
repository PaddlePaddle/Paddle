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

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class FcOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op) override {
    VLOG(4) << "convert a fluid fc op to tensorrt fc layer without bias";

    framework::OpDesc op_desc(op, nullptr, nullptr);

    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);

    // Create weights
    TensorRTEngine::Weight(nvinfer1::DataType::kFLOAT, )


    auto* layer = TRT_ENGINE_ADD_LAYER(
                                       engine_, FullyConnected, *const_cast<nvinfer1::ITensor*>(input1), 
)
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
