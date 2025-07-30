/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/plugin/elementwiseadd_transpose_op_plugin.h"

#include <vector>

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle::inference::tensorrt {
class ElementwiseaddTransposeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a fuse_elementwiseadd_transpose op to tensorrt "
               "elementwiseadd_transpose plugin";
    framework::OpDesc op_desc(op, nullptr);
    auto* input_x = engine_->GetITensor(op_desc.Input("X").front());
    auto* input_y = engine_->GetITensor(op_desc.Input("Y").front());
    std::vector<nvinfer1::ITensor*> inputs{input_x, input_y};
    int axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    std::vector<int> output_shape =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("output_shape"));
    plugin::ElementwiseAddTransposePluginDynamic* plugin =
        new plugin::ElementwiseAddTransposePluginDynamic(axis, output_shape);
    nvinfer1::ILayer* elementwise_layer =
        engine_->AddDynamicPlugin(inputs.data(), 2, plugin);
    std::vector<std::string> output_names;
    output_names.emplace_back(op_desc.Output("Out").front());
    ReplenishLayerAndOutput(elementwise_layer,
                            "fuse_elementwiseadd_transpose",
                            output_names,
                            test_mode);
  }
};

}  // namespace paddle::inference::tensorrt
REGISTER_TRT_OP_CONVERTER(fuse_eleadd_transpose,
                          ElementwiseaddTransposeOpConverter);
