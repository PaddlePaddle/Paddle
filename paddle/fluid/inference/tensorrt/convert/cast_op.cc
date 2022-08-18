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
#include "paddle/fluid/inference/tensorrt/plugin/cast_op_plugin.h"

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

class CastOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a cast op to tensorrt";
    framework::OpDesc op_desc(op, nullptr);

    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto in_dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("in_dtype"));
    auto out_dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("out_dtype"));

    nvinfer1::ILayer* layer = nullptr;
      if (engine_->with_dynamic_shape()) {
        plugin::CastPluginDynamic* plugin =
            new plugin::CastPluginDynamic(in_dtype, out_dtype);
        layer = engine_->AddDynamicPlugin(&input, 1, plugin);
      } else {
        plugin::CastPlugin* plugin =
            new plugin::CastPlugin(in_dtype, out_dtype);
        layer = engine_->AddPlugin(&input, 1, plugin);
      }

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "cast", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(cast, CastOpConverter);
