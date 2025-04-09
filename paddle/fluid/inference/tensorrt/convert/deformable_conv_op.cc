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

#include <cstdio>
#include <vector>

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/deformable_conv_op_plugin.h"

namespace paddle::inference::tensorrt {

class DeformableConvOpConverter : public OpConverter {
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a deformable conv op to tensorrt plugin";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("Input").front();
    std::string offset_name = op_desc.Input("Offset").front();
    std::string mask_name = op_desc.Input("Mask").front();
    std::string filter_name = op_desc.Input("Filter").front();

    auto* input_tensor = engine_->GetITensor(input_name);
    auto* offset_tensor = engine_->GetITensor(offset_name);
    auto* mask_tensor = engine_->GetITensor(mask_name);
    auto* filter_var = scope.FindVar(filter_name);
    auto* filter_tensor = filter_var->GetMutable<phi::DenseTensor>();

    const int c_o = filter_tensor->dims()[0];
    const int c_i = filter_tensor->dims()[1];
    const int k_h = filter_tensor->dims()[2];
    const int k_w = filter_tensor->dims()[3];
    std::vector<int> kernel_dims = {c_o, c_i, k_h, k_w};

    auto strides =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
    auto paddings =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
    auto dilations =
        PADDLE_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));

    auto groups = PADDLE_GET_CONST(int, op_desc.GetAttr("groups"));
    auto deformable_groups =
        PADDLE_GET_CONST(int, op_desc.GetAttr("deformable_groups"));
    auto im2col_step = PADDLE_GET_CONST(int, op_desc.GetAttr("im2col_step"));

    nvinfer1::Weights weights = {};
    weights.count = filter_tensor->numel();
    // TODO(bukejiyu): deformable_conv currently does not support fp16
    // mode,will be supported in the future.
    bool with_fp16 = false;
    if (with_fp16) {
      auto filter_weight = engine_->GetTrtWeight(filter_name, *filter_tensor);
      if (filter_weight.get().type == nvinfer1::DataType::kFLOAT) {
        auto half_filter_data = new half[filter_tensor->numel()];
        for (int i = 0; i < filter_tensor->numel(); i++) {
          half_filter_data[i] = static_cast<half>(
              static_cast<const float*>(filter_weight.get().values)[i]);
        }
        weights.type = nvinfer1::DataType::kHALF;
        weights.values = half_filter_data;
      } else if (filter_weight.get().type == nvinfer1::DataType::kHALF) {
        weights = filter_weight.get();
      }
    } else {
      weights = engine_->GetFp32TrtWeight(filter_name, *filter_tensor).get();
    }
    auto* deformable_conv_plugin = new plugin::DeformableConvPluginDynamic(
        with_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
        weights,
        kernel_dims,
        strides,
        paddings,
        dilations,
        groups,
        deformable_groups,
        im2col_step,
        with_fp16);

    std::vector<nvinfer1::ITensor*> deformable_conv_inputs;
    deformable_conv_inputs.push_back(input_tensor);
    deformable_conv_inputs.push_back(offset_tensor);
    deformable_conv_inputs.push_back(mask_tensor);

    auto* deformable_conv_layer =
        engine_->network()->addPluginV2(deformable_conv_inputs.data(),
                                        deformable_conv_inputs.size(),
                                        *deformable_conv_plugin);

    std::vector<std::string> output_names;
    output_names.push_back(op_desc.Output("Output").front());

    ReplenishLayerAndOutput(
        deformable_conv_layer, "deformable_conv", output_names, test_mode);
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(deformable_conv, DeformableConvOpConverter);
