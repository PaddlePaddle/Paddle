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
#include "paddle/fluid/inference/tensorrt/plugin/prelnlayernorm_shift_partition_op.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class PrelnLayerNormShiftPartitionOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a preln_layernorm_shift_partition op to tensorrt "
               "preln_layernorm_shift_partition plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Y = engine_->GetITensor(op_desc.Input("Y").front());

    std::vector<nvinfer1::ITensor*> inputs{X, Y};

    auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
    auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());

    const float eps = op_desc.HasAttr("epsilon")
                          ? PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"))
                          : 1e-5f;
    const int window_size =
        PADDLE_GET_CONST(int, op_desc.GetAttr("window_size"));
    const int input_resolution =
        PADDLE_GET_CONST(int, op_desc.GetAttr("input_resolution"));

    const int shift_size =
        op_desc.HasAttr("shift_size")
            ? PADDLE_GET_CONST(int, op_desc.GetAttr("shift_size"))
            : 0;

    auto* Bias_t = Bias_v->GetMutable<phi::DenseTensor>();
    auto* Scale_t = Scale_v->GetMutable<phi::DenseTensor>();

    auto bias_weight =
        engine_->GetFp32TrtWeight(op_desc.Input("Bias").front(), *Bias_t);
    auto scale_weight =
        engine_->GetFp32TrtWeight(op_desc.Input("Scale").front(), *Scale_t);
    bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
    nvinfer1::ILayer* layernorm_layer = nullptr;
    plugin::PrelnLnormShiftPartitionPluginDynamic* plugin =
        new plugin::PrelnLnormShiftPartitionPluginDynamic(
            static_cast<const float*>(scale_weight.get().values),
            static_cast<const float*>(bias_weight.get().values),
            bias_weight.get().count,
            shift_size,
            window_size,
            input_resolution,
            eps,
            with_fp16);
    layernorm_layer =
        engine_->AddDynamicPlugin(inputs.data(), inputs.size(), plugin);

    std::vector<std::string> output_names;
    output_names.emplace_back(op_desc.Output("Out_0").front());
    output_names.emplace_back(op_desc.Output("Out_1").front());
    ReplenishLayerAndOutput(layernorm_layer,
                            "preln_layernorm_shift_partition",
                            output_names,
                            test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(preln_layernorm_shift_partition,
                          PrelnLayerNormShiftPartitionOpConverter);
