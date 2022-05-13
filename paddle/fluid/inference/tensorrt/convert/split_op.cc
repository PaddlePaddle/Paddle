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
    size_t input_num = op_desc.Input("X").size();
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
    if (engine_->with_dynamic_shape()) {
      nvinfer1::Dims trt_step_dims;
      trt_step_dims.nbDims = input->getDimensions().nbDims;
      for (int i = 0; i < trt_step_dims.nbDims; i++) trt_step_dims.d[i] = 1;
      auto shape_tensor = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);

      std::vector<int32_t> gather_indices;
      gather_indices.resize(trt_step_dims.nbDims);
      std::iota(gather_indices.begin(), gather_indices.end(), 0);
      gather_indices[axis] = gather_indices.size();
      std::string s = "";
      auto gather_indices_tensor = Add1DConstantLayer(gather_indices, s + "_add_split_op_" + "gather_indices");
      std::vector<int32_t> zeros(trt_step_dims.nbDims, 0);
      auto zeros_tensor = Add1DConstantLayer(zeros, s + "_add_split_op_" + "zeros");

      std::vector<int32_t> each_len(1, output_lengths[0]);
      auto each_len_tensor = Add1DConstantLayer(each_len, s + "_add_split_op_" + "each_len");

      // input : [N,C,H,W]
      for (size_t i = 0; i < output_num; i++) {
        std::vector<int32_t> i_vec(1, i);
        auto i_tensor = Add1DConstantLayer(i_vec, s + "_add_split_op_" + "i");

        auto start_tensor = TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *i_tensor, *each_len_tensor,
            nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
        std::vector<nvinfer1::ITensor*> concat_inputs1 = {zeros_tensor, start_tensor};
        std::vector<nvinfer1::ITensor*> concat_inputs2 = {shape_tensor, each_len_tensor};
        start_tensor = TRT_ENGINE_ADD_LAYER(engine_, Gather,
                 *zeros_tensor,
                *TRT_ENGINE_ADD_LAYER(engine_, Concatenation, concat_inputs1.data() ,2)->getOutput(0),
                0)->getOutput(0);
        auto size_tensor = TRT_ENGINE_ADD_LAYER(engine_, Gather,
                 *shape_tensor,
                *TRT_ENGINE_ADD_LAYER(engine_, Concatenation, concat_inputs2.data() ,2)->getOutput(0),
                0)->getOutput(0);

        layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, trt_step_dims,
                                     trt_step_dims, trt_step_dims);
        layer->setInput(1, *start_tensor);
        layer->setInput(2, *size_tensor);
        
        
        auto output_name = op_desc.Output("Out")[i];
        layer->getOutput(0)->setName(output_name.c_str());
        engine_->SetITensor(output_name, layer->getOutput(0));
        std::string layer_name = "split (Output: ";
        layer_name += output_name;
        if (test_mode) {
          engine_->DeclareOutput(output_name);
        }
        layer->setName((layer_name + ")").c_str());
      }

    } else {
#if IS_TRT_VERSION_GE(7130)

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
        layer->getOutput(i)->setName(output_name.c_str());
        engine_->SetITensor(output_name, layer->getOutput(i));
        std::string layer_name = "split (Output: ";
        layer_name += output_name;
        if (test_mode) {
          engine_->DeclareOutput(output_name);
        }
        layer->setName((layer_name + ")").c_str());
      }
#else
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::SplitPlugin* plugin =
          new plugin::SplitPlugin(axis, output_lengths, with_fp16);
      layer = engine_->AddPluginV2Ext(&input, input_num, plugin);

      std::string layer_name = "split (Output: ";
      for (size_t i = 0; i < output_num; i++) {
        auto output_name = op_desc.Output("Out")[i];
        layer->getOutput(i)->setName(output_name.c_str());
        engine_->SetITensor(output_name, layer->getOutput(i));
        layer_name += output_name;
        if (test_mode) {
          engine_->DeclareOutput(output_name);
        }
      }
      layer->setName((layer_name + ")").c_str());
#endif
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(split, SplitOpConverter);
