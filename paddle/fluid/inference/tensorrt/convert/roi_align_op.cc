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
#include "paddle/fluid/inference/tensorrt/plugin/roi_align_op_plugin.h"

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

/*
 * Roi Align Op
 */
class RoiAlignOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid roi align op to tensorrt plugin";

    framework::OpDesc op_desc(op, nullptr);
    std::string input_name = op_desc.Input("X").front();
    std::string rois_name = op_desc.Input("ROIs").front();
    std::string output_name = op_desc.Output("Out").front();

    const auto pooled_height =
        BOOST_GET_CONST(int, op_desc.GetAttr("pooled_height"));
    const auto pooled_width =
        BOOST_GET_CONST(int, op_desc.GetAttr("pooled_width"));
    const auto spatial_scale =
        BOOST_GET_CONST(float, op_desc.GetAttr("spatial_scale"));
    const auto sampling_ratio =
        BOOST_GET_CONST(int, op_desc.GetAttr("sampling_ratio"));
    const auto aligned = BOOST_GET_CONST(bool, op_desc.GetAttr("aligned"));

    const auto input_tensor = engine_->GetITensor(input_name);
    const auto rois_tensor = engine_->GetITensor(rois_name);

    const nvinfer1::DataType data_type_ = engine_->WithFp16()
                                              ? nvinfer1::DataType::kHALF
                                              : nvinfer1::DataType::kFLOAT;

    std::vector<nvinfer1::ITensor*> inputs{input_tensor, rois_tensor};
    nvinfer1::ILayer* layer = nullptr;

    auto* roi_align_plugin = new plugin::RoiAlignPluginDynamic(
        data_type_, pooled_height, pooled_width, spatial_scale, sampling_ratio,
        aligned);
    auto roi_align_layer = engine_->network()->addPluginV2(
        inputs.data(), inputs.size(), *roi_align_plugin);
    layer = roi_align_layer;

    std::vector<std::string> output_names{output_name};
    RreplenishLayerAndOutput(layer, "roi_align", output_names, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(roi_align, RoiAlignOpConverter);
