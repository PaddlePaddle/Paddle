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
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/fluid/inference/tensorrt/plugin/ms_deform_attn_op_plugin.h"

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

class MsDeformAttnOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
framework::OpDesc op_desc(op, nullptr);
auto* value = engine_->GetITensor(op_desc.Input("value")[0]);
auto* sampling_locations = engine_->GetITensor(op_desc.Input("sampling_locations")[0]);
auto* attention_weights = engine_->GetITensor(op_desc.Input("attention_weights")[0]);
auto* spatial_shapes = engine_->GetITensor(op_desc.Input("spatial_shapes")[0]);
auto* level_start_index = engine_->GetITensor(op_desc.Input("level_start_index")[0]);
int im2col_step = PADDLE_GET_CONST(int, op_desc.GetAttr("im2col_step"));
bool with_fp16 = false;
std::vector<nvinfer1::ITensor*> inputs{value, sampling_locations, attention_weights, spatial_shapes, level_start_index};

    if (engine_->with_dynamic_shape()) {
      plugin::MsDeformAttnPluginDynamic* plugin =
          new plugin::MsDeformAttnPluginDynamic(im2col_step, with_fp16);
      nvinfer1::ILayer* layer =
          engine_->AddDynamicPlugin(inputs.data(), 5, plugin);
      auto output_name = op_desc.Output("out").front();
      ReplenishLayerAndOutput(layer,
                               "ms_deform_attn",
                               {output_name},
                               test_mode);
    }
  }
};


}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(ms_deform_attn, MsDeformAttnOpConverter);

