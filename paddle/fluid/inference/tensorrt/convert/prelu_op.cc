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
#include "paddle/fluid/inference/tensorrt/plugin/prelu_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * PRelu converter from fluid to tensorRT.
 */
class PReluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid prelu op to tensorrt prelu layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
<<<<<<< HEAD
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    auto input_dims = input->getDimensions();
=======
    size_t input_num = op_desc.Input("X").size();
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    // Get attrs
    std::string mode = PADDLE_GET_CONST(std::string, op_desc.GetAttr("mode"));
    std::string data_format = "NCHW";
    if (op_desc.HasAttr("data_format")) {
      data_format =
          PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_format"));
    }
<<<<<<< HEAD

    auto* alpha_var = scope.FindVar(op_desc.Input("Alpha")[0]);
    auto* alpha_weight = alpha_var->GetMutable<phi::DenseTensor>();
    auto w_dims = alpha_weight->dims();
    auto alpha_data =
        engine_->GetFp32TrtWeight(op_desc.Input("Alpha")[0], *alpha_weight);

    nvinfer1::Dims trt_w_dims;
    trt_w_dims.nbDims = w_dims.size();
    for (int i = 0; i < trt_w_dims.nbDims; i++) {
      trt_w_dims.d[i] = w_dims[i];
    }

    // The `element` or `channel` mode contains the batch using static shape.
    if ((mode == "element" || mode == "channel") &&
        !engine_->with_dynamic_shape() &&
        (trt_w_dims.nbDims - 1 == input_dims.nbDims)) {
      trt_w_dims.nbDims--;
      for (int i = 0; i < trt_w_dims.nbDims; i++) {
        trt_w_dims.d[i] = trt_w_dims.d[i + 1];
      }
    }

    nvinfer1::ITensor* alpha_tensor =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, trt_w_dims, alpha_data.get())
            ->getOutput(0);

    auto alpha_dims = alpha_tensor->getDimensions();
    nvinfer1::ITensor* real_alpha_tensor = alpha_tensor;
    if (alpha_dims.nbDims != input_dims.nbDims) {
      auto* reshape_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *alpha_tensor);
      int c = alpha_dims.d[0];
      if (engine_->with_dynamic_shape()) {
        std::vector<nvinfer1::ITensor*> itensors;
        auto* n_tensor = Add1DConstantLayer(1);
        auto* c_tensor = Add1DConstantLayer(c);
        nvinfer1::ITensor* hw_tensor = nullptr;
        nvinfer1::ITensor* shape_tensor = nullptr;
        if (input_dims.nbDims - 2 > 0) {
          hw_tensor = Add1DConstantLayer(
              std::vector<int32_t>(input_dims.nbDims - 2, 1));
        }
        if (data_format == "NCHW") {
          if (hw_tensor != nullptr) {
            shape_tensor = Concat(
                std::vector<nvinfer1::ITensor*>{n_tensor, c_tensor, hw_tensor});

          } else {
            shape_tensor =
                Concat(std::vector<nvinfer1::ITensor*>{n_tensor, c_tensor});
          }
        } else {
          if (hw_tensor != nullptr) {
            shape_tensor = Concat(
                std::vector<nvinfer1::ITensor*>{n_tensor, hw_tensor, c_tensor});
          } else {
            shape_tensor =
                Concat(std::vector<nvinfer1::ITensor*>{n_tensor, c_tensor});
          }
        }
        reshape_layer->setInput(1, *shape_tensor);
      } else {
        nvinfer1::Dims reshape_dim;
        reshape_dim.nbDims = input_dims.nbDims;
        std::fill(reshape_dim.d, reshape_dim.d + input_dims.nbDims, 1);
        if (data_format == "NCHW") {
          reshape_dim.d[0] = c;
        } else if (data_format == "NHWC") {
          reshape_dim.d[input_dims.nbDims - 1] = c;
        }
        reshape_layer->setReshapeDimensions(reshape_dim);
      }
      real_alpha_tensor = reshape_layer->getOutput(0);
    }

    nvinfer1::ILayer* layer = nullptr;

    layer = TRT_ENGINE_ADD_LAYER(
        engine_, ParametricReLU, *input, *real_alpha_tensor);
=======
    auto* alpha_var = scope.FindVar(op_desc.Input("Alpha")[0]);
    auto* alpha_tensor = alpha_var->GetMutable<framework::LoDTensor>();

    auto alpha_weight =
        engine_->GetFp32TrtWeight(op_desc.Input("Alpha")[0], *alpha_tensor);

    platform::CPUPlace cpu_place;

    nvinfer1::ILayer* layer = nullptr;
    if (engine_->with_dynamic_shape()) {
      plugin::PReluPluginDynamic* plugin = new plugin::PReluPluginDynamic(
          static_cast<const float*>(alpha_weight.get().values),
          alpha_tensor->numel(),
          mode,
          data_format);
      layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
    } else {
#if IS_TRT_VERSION_GE(7000)
      nvinfer1::Dims dims;
      dims.nbDims = 0;
      // jump batch dim
      for (int i = 1; i < alpha_tensor->dims().size(); i++) {
        dims.d[dims.nbDims++] = alpha_tensor->dims()[i];
      }
      for (; dims.nbDims < input->getDimensions().nbDims; dims.nbDims++) {
        dims.d[dims.nbDims] = 1;
      }

      auto alpha_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, dims, alpha_weight.get());
      auto alpha_layer_output = alpha_layer->getOutput(0);

      layer = TRT_ENGINE_ADD_LAYER(
          engine_, ParametricReLU, *input, *alpha_layer_output);
#else
      plugin::PReluPlugin* plugin = new plugin::PReluPlugin(
          static_cast<const float*>(alpha_weight.get().values),
          alpha_tensor->numel(),
          mode,
          data_format);
      layer = engine_->AddPlugin(&input, input_num, plugin);
#endif
    }
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "prelu", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(prelu, PReluOpConverter);
