/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/inference/tensorrt/plugin/yolo_box_head_op_plugin.h"

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework
namespace paddle::framework::proto {
class OpDesc;
}  // namespace paddle::framework::proto

namespace paddle::inference::tensorrt {

class YoloBoxHeadOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a yolo_box_head op to tensorrt plugin";

    framework::OpDesc op_desc(op, nullptr);
    auto* x_tensor = engine_->GetITensor(op_desc.Input("X").front());
    std::vector<int> anchors =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("anchors"));
    int class_num = PADDLE_GET_CONST(int, op_desc.GetAttr("class_num"));

    auto* yolo_box_plugin = new plugin::YoloBoxHeadPlugin(anchors, class_num);
    std::vector<nvinfer1::ITensor*> yolo_box_inputs;
    yolo_box_inputs.push_back(x_tensor);
    auto* yolo_box_head_layer = engine_->network()->addPluginV2(
        yolo_box_inputs.data(), yolo_box_inputs.size(), *yolo_box_plugin);
    std::vector<std::string> output_names;
    output_names.push_back(op_desc.Output("Out").front());
    ReplenishLayerAndOutput(
        yolo_box_head_layer, "yolo_box_head", output_names, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(yolo_box_head, YoloBoxHeadOpConverter);
