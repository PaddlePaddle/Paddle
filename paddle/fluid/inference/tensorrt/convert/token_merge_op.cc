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
#include "paddle/fluid/inference/tensorrt/plugin/token_merge_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class TokenMergeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a token_merge op to tensorrt "
               "token_merge plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* X = engine_->GetITensor(op_desc.Input("origined_tensor").front());
    float ratio = PADDLE_GET_CONST(float, op_desc.GetAttr("ratio"));
    bool use_rand = PADDLE_GET_CONST(bool, op_desc.GetAttr("use_rand"));
  
    nvinfer1::Dims dims_x = X->getDimensions();

    int bsz = dims_x.d[0];
    int token_number = dims_x.d[1];
    int hid_dim = dims_x.d[2];

    nvinfer1::ILayer* token_merge_layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::TokenMergePluginDynamic* plugin =
          new plugin::TokenMergePluginDynamic(
              with_fp16,
              bsz,
              token_number,
              hid_dim,
              ratio,
              use_rand);
      token_merge_layer = engine_->AddDynamicPlugin(&X, 1, plugin);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "token_merge do not support static shape mode yet"));
    }

    auto output_merged_tensor_name = op_desc.Output("merged_tensor").front();
    auto output_rand_select_arr_name = op_desc.Output("rand_select_arr").front();
    auto output_whether_tobe_merge_name = op_desc.Output("whether_tobe_merge").front();
    RreplenishLayerAndOutput(token_merge_layer,
                             "token_merge",
                             {output_merged_tensor_name, output_rand_select_arr_name, output_whether_tobe_merge_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(token_merge, TokenMergeOpConverter);