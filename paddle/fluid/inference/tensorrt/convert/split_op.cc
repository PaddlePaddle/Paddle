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
#include "paddle/fluid/inference/tensorrt/plugin/split_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SplitOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert a fluid split op to tensorrt split layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
    size_t output_num = op_desc.Output("Out").size();

    // Get Attrs
    int axis = BOOST_GET_CONST(int, op_desc.GetAttr("axis"));

    std::vector<int> output_lengths =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("sections"));
    int num = 0;
    if (op_desc.HasAttr("num")) {
      num = BOOST_GET_CONST(int, op_desc.GetAttr("num"));
    }

    if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
      axis += (axis < 0) ? input_dims.nbDims : 0;
#endif
    } else {
      axis += (axis < 0) ? input_dims.nbDims : -1;
    }
    if (num > 0) {
      int64_t in_axis_dim = input_dims.d[axis];
      size_t out_axis_dim = in_axis_dim / num;
      for (int i = 0; i < num; ++i) {
        output_lengths.push_back(out_axis_dim);
      }
    }
    nvinfer1::ILayer* layer = nullptr;
#if IS_TRT_VERSION_GE(8034)
    if (engine_->with_dynamic_shape()) {
      nvinfer1::Dims trt_step_dims;
      trt_step_dims.nbDims = input->getDimensions().nbDims;
      for (int i = 0; i < trt_step_dims.nbDims; i++) trt_step_dims.d[i] = 1;

      auto shape_tensor =
          TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);

      std::vector<int32_t> gather_indices;
      gather_indices.resize(trt_step_dims.nbDims);
      std::iota(gather_indices.begin(), gather_indices.end(), 0);
      gather_indices[axis] = gather_indices.size();
      std::string name = "_add_split_op_";
      auto gather_indices_tensor =
          Add1DConstantLayer(gather_indices, name + "gather_indices");
      std::vector<int32_t> zeros(trt_step_dims.nbDims, 0);
      auto zeros_tensor = Add1DConstantLayer(zeros, name + "zeros");
      // input : [N,C,H,W]
      for (size_t i = 0; i < output_num; i++) {
        auto this_len_tensor =
            Add1DConstantLayer(output_lengths[i], name + "this_len_tensor");
        auto start_point_tensor =
            Add1DConstantLayer(std::accumulate(output_lengths.begin(),
                                               output_lengths.begin() + i, 0),
                               name + "start_point_tensor");

        std::vector<nvinfer1::ITensor*> concat_inputs1 = {zeros_tensor,
                                                          start_point_tensor};
        std::vector<nvinfer1::ITensor*> concat_inputs2 = {shape_tensor,
                                                          this_len_tensor};
        auto start_tensor =
            TRT_ENGINE_ADD_LAYER(engine_, Gather,
                                 *TRT_ENGINE_ADD_LAYER(engine_, Concatenation,
                                                       concat_inputs1.data(), 2)
                                      ->getOutput(0),
                                 *gather_indices_tensor, 0)
                ->getOutput(0);
        auto size_tensor =
            TRT_ENGINE_ADD_LAYER(engine_, Gather,
                                 *TRT_ENGINE_ADD_LAYER(engine_, Concatenation,
                                                       concat_inputs2.data(), 2)
                                      ->getOutput(0),
                                 *gather_indices_tensor, 0)
                ->getOutput(0);

        layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, trt_step_dims,
                                     trt_step_dims, trt_step_dims);
        layer->setInput(1, *start_tensor);
        layer->setInput(2, *size_tensor);

        auto output_name = op_desc.Output("Out")[i];
        RreplenishLayerAndOutput(layer, "split", {output_name}, test_mode);
      }
    } else {
      auto chw_input_dims = input->getDimensions();
      nvinfer1::Dims trt_start_dims;
      trt_start_dims.nbDims = chw_input_dims.nbDims;
      memset(trt_start_dims.d, 0, sizeof(int32_t) * chw_input_dims.nbDims);
      nvinfer1::Dims trt_size_dims = chw_input_dims;
      nvinfer1::Dims trt_step_dims;
      trt_step_dims.nbDims = chw_input_dims.nbDims;
      for (int i = 0; i < trt_step_dims.nbDims; i++) trt_step_dims.d[i] = 1;

      // input : [C,H,W]
      for (size_t i = 0; i < output_num; i++) {
        trt_start_dims.d[axis] = std::accumulate(output_lengths.begin(),
                                                 output_lengths.begin() + i, 0);
        trt_size_dims.d[axis] = output_lengths[i];
        layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, trt_start_dims,
                                     trt_size_dims, trt_step_dims);
        auto output_name = op_desc.Output("Out")[i];
        RreplenishLayerAndOutput(layer, "split", {output_name}, test_mode);
      }
    }
#else
    if (engine_->with_dynamic_shape()) {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::SplitPluginDynamic* plugin =
          new plugin::SplitPluginDynamic(axis, output_lengths, with_fp16);
      layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
    } else {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::SplitPlugin* plugin =
          new plugin::SplitPlugin(axis, output_lengths, with_fp16);
      layer = engine_->AddPluginV2Ext(&input, input_num, plugin);
    }
    std::string output_names;
    for (size_t i = 0; i < output_num; i++) {
      output_names += op_desc.Output("Out")[i];
    }
    RreplenishLayerAndOutput(layer, "split", output_names, test_mode);
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(split, SplitOpConverter);
