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
#include "paddle/fluid/inference/tensorrt/plugin/transformer_input_output_convert_plugin.h"

namespace paddle::inference::tensorrt {

/*
 * Convert Transformer Input(pos_id, max_seqlen).
 */
class TransformerInputConvert : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "Convert Transformer Input(pos_id, max_seqlen), use "
               "transformer_input_convert_plugin";
    if (!engine_->with_dynamic_shape()) {
      PADDLE_THROW(common::errors::Fatal(
          "transformer_input_convert_op: If you want to use transformer, must "
          "be with dynamic shape"));
    }

    framework::OpDesc op_desc(op, nullptr);
    auto input_name = op_desc.Input("Input").front();
    auto* input = engine_->GetITensor(input_name);
    int input_num = op_desc.Input("Input").size();

    // tensorrt_subgraph_pass will rename tensor
    // auto pos_id_name = op_desc.Output("PosId").front();
    // auto max_seqlen_name = op_desc.Output("MaxSeqlen").front();
    auto pos_id_name = "pos_id_tensor";
    auto max_seqlen_name = "max_seqlen_tensor";

    plugin::TransformerInputConvertPlugin* plugin =
        new plugin::TransformerInputConvertPlugin();
    nvinfer1::ILayer* layer =
        engine_->AddDynamicPlugin(&input, input_num, plugin);

    ReplenishLayerAndOutput(layer,
                            "transformer_input_convert",
                            {pos_id_name, max_seqlen_name},
                            test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(transformer_input_convert, TransformerInputConvert);
