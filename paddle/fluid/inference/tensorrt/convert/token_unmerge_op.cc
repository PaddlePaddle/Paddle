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
#include "paddle/fluid/inference/tensorrt/plugin/token_unmerge_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class TokenUnmergeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert a token_unmerge op to tensorrt "
               "token_unmerge plugin";
    framework::OpDesc op_desc(op, nullptr);

    auto* merged_tensor = engine_->GetITensor(op_desc.Input("merged_tensor").front());
    auto* rand_select_arr = engine_->GetITensor(op_desc.Input("rand_select_arr").front());
    auto* whether_tobe_merge = engine_->GetITensor(op_desc.Input("whether_tobe_merge").front());
    int token_number = PADDLE_GET_CONST(int, op_desc.GetAttr("token_number"));
  
    nvinfer1::Dims dims_x = merged_tensor->getDimensions();

    int bsz = dims_x.d[0];
    int final_token_number = dims_x.d[1];
    int hid_dim = dims_x.d[2];

    nvinfer1::ILayer* token_unmerge_layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      VLOG(4) << "create a"
               "token_unmerge plugin, hid_dim = " << hid_dim;
      plugin::TokenUnmergePluginDynamic* plugin =
          new plugin::TokenUnmergePluginDynamic(with_fp16,
          bsz,
          token_number,
          final_token_number,
          hid_dim);

      std::vector<nvinfer1::ITensor*> inputs = {
        merged_tensor,
        rand_select_arr,
        whether_tobe_merge};
      token_unmerge_layer = engine_->AddDynamicPlugin(inputs.data(), inputs.size(), plugin);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "token_unmerge do not support static shape mode yet"));
    }

    auto output_unmerged_tensor_name = op_desc.Output("unmerged_tensor").front();
    RreplenishLayerAndOutput(token_unmerge_layer,
                             "token_unmerge",
                             {output_unmerged_tensor_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(token_unmerge, TokenUnmergeOpConverter);