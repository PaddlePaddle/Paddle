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
#include "paddle/fluid/inference/tensorrt/plugin/fused_token_prune_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class FusedTokenPruneOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;

    auto* Attn = engine_->GetITensor(op_desc.Input("Attn").front());
    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Mask = engine_->GetITensor(op_desc.Input("Mask").front());
    auto* NewMask = engine_->GetITensor(op_desc.Input("NewMask").front());
    bool keep_first_token =
        op_desc.HasAttr("keep_first_token")
            ? BOOST_GET_CONST(bool, op_desc.GetAttr("keep_first_token"))
            : true;
    bool keep_order = op_desc.HasAttr("keep_order")
                          ? BOOST_GET_CONST(bool, op_desc.GetAttr("keep_order"))
                          : false;

    std::vector<nvinfer1::ITensor*> itensors = {Attn, X, Mask, NewMask};

    auto output_name = op_desc.Output("SlimmedX")[0];
    auto out_inds_name = op_desc.Output("CLSInds")[0];
    if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();

      if (engine_->precision() == AnalysisConfig::Precision::kInt8) {
        with_fp16 = true;
      }
      plugin::FusedTokenPrunePluginDynamic* plugin =
          new plugin::FusedTokenPrunePluginDynamic(
              with_fp16, keep_first_token, keep_order);
      layer = engine_->AddDynamicPlugin(itensors.data(), 4, plugin);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the TRT Dynamic Shape mode, need to confirm that "
          "your TRT version is no less than 6.0"));
#endif
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the Ernie(Bert) model in static shape mode, which "
          "is not supported for the time being.\n"
          "You can use the config.SetTRTDynamicShapeInfo(...) interface to set "
          "the shape information to run the dynamic shape mode."));
    }
    RreplenishLayerAndOutput(
        layer, "fused_token_prune", {output_name, out_inds_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fused_token_prune, FusedTokenPruneOpConverter);
