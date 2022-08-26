/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/plugin/group_norm_op_plugin.h"

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

#include <vector>

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

class GroupNormOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid group_norm op";

    framework::OpDesc op_desc(op, nullptr);

    auto* input_itensor = engine_->GetITensor(op_desc.Input("X").front());

    int groups = PADDLE_GET_CONST(int, op_desc.GetAttr("groups"));
    float epsilon = PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"));

    std::string scale_name = op_desc.Input("Scale").front();
    std::string bias_name = op_desc.Input("Bias").front();

    // get the presistable var's data
    auto GetWeight = [&](const std::string& var_name,
                         framework::DDim* dims) -> TensorRTEngine::Weight {
      auto* temp_var = scope.FindVar(var_name);
      auto* temp_tensor = temp_var->GetMutable<framework::LoDTensor>();
      (*dims) = temp_tensor->dims();

      auto weight = engine_->GetTrtWeight(var_name, *temp_tensor);
      return weight;
    };

    framework::DDim scale_dims;
    framework::DDim bias_dims;
    auto scale_weights = GetWeight(scale_name, &scale_dims);
    auto bias_weights = GetWeight(bias_name, &bias_dims);
    if (engine_->with_dynamic_shape()) {
      int gn_num = groups;
      std::vector<int64_t> mean_shape({gn_num});
      std::vector<int64_t> variance_shape({gn_num});
      plugin::GroupNormPluginDynamic* plugin =
          new plugin::GroupNormPluginDynamic(
              static_cast<const float*>(scale_weights.get().values),
              scale_weights.get().count,
              static_cast<const float*>(bias_weights.get().values),
              bias_weights.get().count,
              epsilon,
              groups,
              mean_shape,
              variance_shape);
      nvinfer1::ILayer* groupnorm_layer =
          engine_->AddDynamicPlugin(&input_itensor, 1, plugin);
      auto output_name = op_desc.Output("Y")[0];
      RreplenishLayerAndOutput(
          groupnorm_layer, "group_norm", {output_name}, test_mode);
    } else {
      int gn_num = input_itensor->getDimensions().d[0] * groups;
      std::vector<int64_t> mean_shape({gn_num});
      std::vector<int64_t> variance_shape({gn_num});
      plugin::GroupNormPlugin* plugin = new plugin::GroupNormPlugin(
          static_cast<const float*>(scale_weights.get().values),
          scale_weights.get().count,
          static_cast<const float*>(bias_weights.get().values),
          bias_weights.get().count,
          epsilon,
          groups,
          mean_shape,
          variance_shape);
      nvinfer1::ILayer* groupnorm_layer =
          engine_->AddPlugin(&input_itensor, 1, plugin);
      auto output_name = op_desc.Output("Y")[0];
      RreplenishLayerAndOutput(
          groupnorm_layer, "group_norm", {output_name}, test_mode);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(group_norm, GroupNormOpConverter);
