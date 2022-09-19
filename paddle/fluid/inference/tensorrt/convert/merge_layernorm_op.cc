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

#include "paddle/fluid/inference/tensorrt/plugin/merge_layernorm_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class MergeLayernormOpConverter : public OpConverter {
    public:
    void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
        VLOG(4) << "convert a fluid merge_layernorm op to tensorrt merge_layernorm plugin";
        framework::OpDesc op_desc(op, nullptr);
        auto* X = engine_->GetITensor(op_desc.Input("X").front());
        auto* Bias_v = scope.FindVar(op_desc.Input("Bias").front());
        auto* Scale_v = scope.FindVar(op_desc.Input("Scale").front());
        const int begin_norm_axis =
            op_desc.HasAttr("begin_norm_axis")
                ? PADDLE_GET_CONST(int, op_desc.GetAttr("begin_norm_axis"))
                : 1;
        const float eps = op_desc.HasAttr("epsilon")
                            ? PADDLE_GET_CONST(float, op_desc.GetAttr("epsilon"))
                            : 1e-5f;
        PADDLE_ENFORCE_NOT_NULL(
        Bias_v,
        platform::errors::InvalidArgument(
            "Input(Bias) of layer_norm should not be null."));
        PADDLE_ENFORCE_NOT_NULL(
        Scale_v,
        platform::errors::InvalidArgument(
            "Input(Scale) of layer_norm should not be null."));
        PADDLE_ENFORCE_EQ(
        begin_norm_axis,
        2,
        platform::errors::InvalidArgument(
            "The begin_norm_axis of LayernormShiftPartition should be %d",
            begin_norm_axis));
        auto* Bias_t = Bias_v->GetMutable<framework::LoDTensor>();
        auto* Scale_t = Scale_v->GetMutable<framework::LoDTensor>();

        auto bias_weight =
            engine_->GetFp32TrtWeight(op_desc.Input("Bias").front(), *Bias_t);
        auto scale_weight =
            engine_->GetFp32TrtWeight(op_desc.Input("Scale").front(), *Scale_t);
        bool with_fp16 = engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
        nvinfer1::ILayer* merge_layernorm_layer = nullptr;
        if (engine_->with_dynamic_shape()) {
            plugin::MergeLayernormPluginDynamic* plugin 
                = new plugin::MergeLayernormPluginDynamic(
                    static_cast<const float*>(bias_weight.get().values),
                    bias_weight.get().count,
                    static_cast<const float*>(scale_weight.get().values),
                    scale_weight.get().count,
                    eps,
                    begin_norm_axis,
                    with_fp16);
            merge_layernorm_layer = engine_->AddDynamicPlugin(&X, 1, plugin);
        } else {
            PADDLE_THROW(platform::errors::InvalidArgument(
                "Currently, MergeLayernorm TRT Plugin only support dynamic shape mode."));

        }
        auto output_name = op_desc.Output("Y").front();
        RreplenishLayerAndOutput(
        merge_layernorm_layer, "merge_layernorm", {output_name}, test_mode);

    }
};

} // tensorrt
} // inference
} // paddle

REGISTER_TRT_OP_CONVERTER(merge_layernorm, 
                          MergeLayernormOpConverter);