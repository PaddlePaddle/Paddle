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
#include "paddle/fluid/inference/tensorrt/plugin/c_allreduce_op_plugin.h"

#define RedType paddle::inference::tensorrt::plugin::kRedSum

namespace nvinfer1 {
class ILayer;
}  // namespace nvinfer1
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

class CAllReduceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid callreduce op to tensorrt layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    int input_num = op_desc.Input("X").size();
    PADDLE_ENFORCE_EQ(input_num, 1,
                      platform::errors::InvalidArgument(
                          "The input X's size must equal to 1 in TRT swish op."
                          " But received X's size %d.",
                          input_num));
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    // Get output
    size_t output_num = op_desc.Output("Out").size();
    PADDLE_ENFORCE_EQ(
        output_num, 1UL,
        platform::errors::InvalidArgument(
            "The ouput Out's size must equal to 1 in TRT swish op. "
            "But received Out's size %u.",
            output_num));
    // Get attrs
    int ring_id = BOOST_GET_CONST(int, op_desc.GetAttr("ring_id"));
    bool use_calc_stream =
        BOOST_GET_CONST(bool, op_desc.GetAttr("use_calc_stream"));

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::CAllReducePluginDynamic* plugin =
          new plugin::CAllReducePluginDynamic(ring_id, use_calc_stream, RedType,
                                              with_fp16);
      layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "You are running the TRT Dynamic Shape mode, need to confirm that "
          "your TRT version is no less than 6.0"));
#endif
    } else {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::CAllReducePlugin* plugin = new plugin::CAllReducePlugin(
          ring_id, use_calc_stream, RedType, with_fp16);
      layer = engine_->AddPlugin(&input, input_num, plugin);
    }

    auto output_name = op_desc.Output("Out")[0];
    std::string name = "c_allreduce_sum";
    switch (RedType) {
      case paddle::inference::tensorrt::plugin::kRedSum:
        name = "c_allreduce_sum";
        break;

      case paddle::inference::tensorrt::plugin::kRedMax:
        name = "c_allreduce_max";
        break;

      case paddle::inference::tensorrt::plugin::kRedMin:
        name = "c_allreduce_min";
        break;

      case paddle::inference::tensorrt::plugin::kRedProd:
        name = "c_allreduce_prod";
        break;

      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid reduce type: %d", RedType));
    }
    RreplenishLayerAndOutput(layer, name, {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(c_allreduce_sum, CAllReduceOpConverter);
