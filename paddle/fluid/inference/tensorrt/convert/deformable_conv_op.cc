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

class DeformableConvOpConverter : public OpConverter {
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
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
    auto* filter_tensor = filter_var->GetMutable<framework::LoDTensor>();

    float* filter_data =
        engine_->GetWeightCPUData(filter_name, filter_tensor, false);
    nvinfer1::Weights weights{nvinfer1::DataType::kFLOAT,
                              static_cast<void*>(filter_data),
                              static_cast<int64_t>(filter_tensor->numel())};
    PADDLE_ENFORCE_EQ(
        filter_tensor->dims().size(), 4UL,
        platform::errors::InvalidArgument(
            "The deformable conv filter's dims size should be 4, but got %d",
            filter_tensor->dims().size()));
    const int c_o = filter_tensor->dims()[0];
    const int c_i = filter_tensor->dims()[1];
    const int k_h = filter_tensor->dims()[2];
    const int k_w = filter_tensor->dims()[3];
    std::vector<int> kernel_dims = {c_o, c_i, k_h, k_w};

    auto strides =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("strides"));
    auto paddings =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("paddings"));
    auto dilations =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("dilations"));

    auto groups = BOOST_GET_CONST(int, op_desc.GetAttr("groups"));
    auto deformable_groups =
        BOOST_GET_CONST(int, op_desc.GetAttr("deformable_groups"));
    auto im2col_step = BOOST_GET_CONST(int, op_desc.GetAttr("im2col_step"));

    int type_id = static_cast<int>(engine_->WithFp16());
    auto* deformable_conv_plugin = new plugin::DeformableConvPlugin(
        type_id ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
        weights, kernel_dims, strides, paddings, dilations, groups,
        deformable_groups, im2col_step);

    std::vector<nvinfer1::ITensor*> deformable_conv_inputs;
    deformable_conv_inputs.push_back(input_tensor);
    deformable_conv_inputs.push_back(offset_tensor);
    deformable_conv_inputs.push_back(mask_tensor);

    auto* deformable_conv_layer = engine_->network()->addPluginV2(
        deformable_conv_inputs.data(), deformable_conv_inputs.size(),
        *deformable_conv_plugin);

    std::vector<std::string> output_names;
    output_names.push_back(op_desc.Output("Output").front());

    RreplenishLayerAndOutput(deformable_conv_layer, "deformable_conv",
                             output_names, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(deformable_conv, DeformableConvOpConverter);
