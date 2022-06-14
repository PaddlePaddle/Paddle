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
#include "paddle/fluid/inference/tensorrt/plugin/slice_op_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/special_slice_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SliceOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    // This OP is implemented by trt dynamic shpae plugin.
    // Dynamic shape plugin requires TRT version greater than 6.0.
    VLOG(4) << "convert slice op to tensorrt layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("Input")[0]);
    auto output_name = op_desc.Output("Out")[0];

    float out_scale = 1;
    if (op_desc.HasAttr("out_threshold")) {
      out_scale = BOOST_GET_CONST(float, op_desc.GetAttr("out_threshold"));
      engine_->SetTensorDynamicRange(input, out_scale);
    }

    std::vector<int> axes =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("axes"));
    std::vector<int> starts =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("starts"));
    std::vector<int> ends =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("ends"));
    std::vector<int> decrease_axises =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("decrease_axis"));

    auto input_dims = input->getDimensions();
    if (!engine_->with_dynamic_shape()) {
      // notice that input shape is [CHW] without batch axis when input has
      // static shape
      for (size_t i = input_dims.nbDims; i > 0; i--) {
        input_dims.d[i] = input_dims.d[i - 1];
      }
      input_dims.d[0] = 1;  // fake batchsize, not useful here
      for (size_t i = 0; i < axes.size(); i++) {
        if (starts[i] < 0) {
          starts[i] = std::max(starts[i] + input_dims.d[axes[i]], 0);
        }
        if (ends[i] < 0) {
          ends[i] = std::max(ends[i] + input_dims.d[axes[i]], 0);
        }
        ends[i] = std::min(ends[i], input_dims.d[axes[i]]);
        PADDLE_ENFORCE_GT(
            ends[i], starts[i],
            platform::errors::InvalidArgument(
                "Attr(ends) should be greater than attr(starts) in "
                "slice op. But received ends = %d, starts = %d.",
                ends[i], starts[i]));
      }
    }

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      if (engine_->use_oss() && engine_->with_ernie() &&
          input_dims.nbDims == 4) {
        std::vector<nvinfer1::ITensor*> plugin_inputs;
        if (engine_->with_interleaved()) {
          auto* shuffler_slice = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input);
          nvinfer1::Permutation transpose_embed{2, 1, 0, 3};
          shuffler_slice->setSecondTranspose(transpose_embed);
          engine_->SetTensorDynamicRange(shuffler_slice->getOutput(0),
                                         out_scale);
          shuffler_slice->setName(
              ("SpecialSlice_interleaved: transpose: (Output: " + output_name +
               ")")
                  .c_str());
          plugin_inputs.emplace_back(shuffler_slice->getOutput(0));
        } else {
          plugin_inputs.emplace_back(input);
        }
        std::string pos_name;
        if (engine_->Has("ernie_pos_name")) {
          pos_name = engine_->Get<std::string>("ernie_pos_name");
        } else {
          // hard code for compatibility
          pos_name = engine_->network()->getInput(2)->getName();
        }
        plugin_inputs.emplace_back(
            engine_->GetITensor(pos_name));  // cu_seqlens, eval_placeholder_2

        // bool ban_fp16 = engine_->disable_trt_plugin_fp16();
        plugin::SpecialSlicePluginDynamic* plugin =
            new plugin::SpecialSlicePluginDynamic();
        layer = engine_->AddDynamicPlugin(plugin_inputs.data(),
                                          plugin_inputs.size(), plugin);
      } else {
        bool with_fp16 =
            engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
        int decrease_axis =
            decrease_axises.size() == 0 ? -1 : decrease_axises[0];
        plugin::SlicePluginDynamic* plugin = new plugin::SlicePluginDynamic(
            starts, ends, axes, decrease_axis, with_fp16);
        layer = engine_->AddDynamicPlugin(&input, 1, plugin);
      }
    } else {
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::SlicePlugin* plugin =
          new plugin::SlicePlugin(starts, ends, axes, with_fp16);
      layer = engine_->AddPlugin(&input, 1, plugin);
    }
    RreplenishLayerAndOutput(layer, "slice", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(slice, SliceOpConverter);
