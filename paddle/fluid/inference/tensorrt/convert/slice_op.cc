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
#if IS_TRT_VERSION_GE(8034)
        auto nchw_input_dims = input->getDimensions();
        nvinfer1::Dims trt_start_dims;
        trt_start_dims.nbDims = nchw_input_dims.nbDims;
        memset(trt_start_dims.d, 0, sizeof(int32_t) * nchw_input_dims.nbDims);

        nvinfer1::Dims trt_size_dims;
        trt_size_dims.nbDims = nchw_input_dims.nbDims;

        nvinfer1::Dims trt_end_dims;
        trt_end_dims.nbDims = nchw_input_dims.nbDims;
        for (int i = 0; i < trt_end_dims.nbDims; i++)
          trt_end_dims.d[i] = 10000000;

        nvinfer1::Dims trt_step_dims;
        trt_step_dims.nbDims = nchw_input_dims.nbDims;
        for (int i = 0; i < trt_step_dims.nbDims; i++) trt_step_dims.d[i] = 1;

        // input : [N,C,H,W]
        for (size_t i = 0; i < axes.size(); i++) {
          int trt_axis = axes[i];
          trt_start_dims.d[trt_axis] = starts[i];
          trt_end_dims.d[trt_axis] = ends[i];
          PADDLE_ENFORCE_GE(starts[i], 0,
                            platform::errors::InvalidArgument(
                                "Attr(starts) should be >= 0 in slice op in "
                                "TensorRT dynamic shape,"
                                "but received starts = %d.",
                                starts[i]));
          PADDLE_ENFORCE_GE(ends[i], 0,
                            platform::errors::InvalidArgument(
                                "Attr(ends) should be >= 0 in slice op in "
                                "TensorRT dynamic shape,"
                                "but received ends = %d.",
                                ends[i]));
        }

        auto start_tensor = Add1DConstantLayer(
            trt_start_dims, output_name + "_add_slice_op_" + "starts");
        auto end_tensor = Add1DConstantLayer(
            trt_end_dims, output_name + "_add_slice_op_" + "ends");

        auto shape_tensor =
            TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);
        auto real_end_tensor =
            TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *shape_tensor,
                                 *end_tensor,
                                 nvinfer1::ElementWiseOperation::kMIN)
                ->getOutput(0);

        auto size_tensor =
            TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *real_end_tensor,
                                 *start_tensor,
                                 nvinfer1::ElementWiseOperation::kSUB)
                ->getOutput(0);

        layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, trt_start_dims,
                                     trt_size_dims, trt_step_dims);
        layer->setInput(1, *start_tensor);
        layer->setInput(2, *size_tensor);
        if (decrease_axises.size() > 0) {
          int decrease_axis = decrease_axises[0];
          std::vector<int32_t> gather_indices;
          for (int i = 0; i < trt_size_dims.nbDims; i++) {
            if (i == decrease_axis) continue;
            gather_indices.push_back(i);
          }
          if (gather_indices.empty()) gather_indices.push_back(decrease_axis);
          auto gather_indices_tensor = Add1DConstantLayer(
              gather_indices,
              output_name + "_add_slice_op_" + "gather_indices");
          auto shape_tensor =
              TRT_ENGINE_ADD_LAYER(engine_, Shape, *layer->getOutput(0))
                  ->getOutput(0);
          auto real_size_tensor =
              TRT_ENGINE_ADD_LAYER(engine_, Gather, *shape_tensor,
                                   *gather_indices_tensor, 0)
                  ->getOutput(0);
          layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *layer->getOutput(0));
          layer->setInput(1, *real_size_tensor);
        }

#else
        bool with_fp16 =
            engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
        int decrease_axis =
            decrease_axises.size() == 0 ? -1 : decrease_axises[0];
        plugin::SlicePluginDynamic* plugin = new plugin::SlicePluginDynamic(
            starts, ends, axes, decrease_axis, with_fp16);
        layer = engine_->AddDynamicPlugin(&input, 1, plugin);

#endif
      }
    } else {
#if IS_TRT_VERSION_GE(8034)
      auto chw_input_dims = input->getDimensions();
      nvinfer1::Dims trt_start_dims;
      trt_start_dims.nbDims = chw_input_dims.nbDims;
      memset(trt_start_dims.d, 0, sizeof(int32_t) * chw_input_dims.nbDims);
      nvinfer1::Dims trt_size_dims = chw_input_dims;
      nvinfer1::Dims trt_step_dims;
      trt_step_dims.nbDims = chw_input_dims.nbDims;
      for (int i = 0; i < trt_step_dims.nbDims; i++) trt_step_dims.d[i] = 1;

      // input : [C,H,W]
      for (size_t i = 0; i < axes.size(); i++) {
        int trt_axis = axes[i] - 1;
        trt_start_dims.d[trt_axis] = starts[i];
        trt_size_dims.d[trt_axis] = ends[i] - starts[i];
      }
      layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, trt_start_dims,
                                   trt_size_dims, trt_step_dims);
#else
      bool with_fp16 =
          engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
      plugin::SlicePlugin* plugin =
          new plugin::SlicePlugin(starts, ends, axes, with_fp16);
      layer = engine_->AddPlugin(&input, 1, plugin);
#endif
    }
    RreplenishLayerAndOutput(layer, "slice", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(slice, SliceOpConverter);
