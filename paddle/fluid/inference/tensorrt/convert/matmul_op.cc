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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/matmul_op_int8_plugin.h"

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
 * MatMulOp, IMatrixMultiplyLayer in TRT. This Layer doesn't has weights.
 */
class MatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid matmul op to tensorrt matmul layer ";
    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;

    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);

    nvinfer1::Dims dims_x = input1->getDimensions();
    nvinfer1::Dims dims_y = input2->getDimensions();

    bool transpose_X = BOOST_GET_CONST(bool, op_desc.GetAttr("transpose_X"));
    bool transpose_Y = BOOST_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));

    auto output_name = op_desc.Output("Out")[0];
    float alpha = 1;
    if (op_desc.HasAttr("alpha")) {
      float alpha_tem = BOOST_GET_CONST(float, op_desc.GetAttr("alpha"));
      alpha = alpha_tem;
    }
    nvinfer1::MatrixOperation matrix_operation_X =
        transpose_X ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;
    nvinfer1::MatrixOperation matrix_operation_Y =
        transpose_Y ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;

    if (op_desc.HasAttr("support_int8") &&
        engine_->precision() == AnalysisConfig::Precision::kInt8) {
      if (engine_->with_dynamic_shape()) {
        VLOG(3) << "Convert a fluid matmul_op_int8_dynamic to TensorRT "
                   "MatmulPluginLayer";
        plugin::MatmulPluginDynamic* plugin =
            new plugin::MatmulPluginDynamic(transpose_X, transpose_Y, alpha);
        std::vector<nvinfer1::ITensor*> inputs{input1, input2};
        layer = engine_->AddDynamicPlugin(inputs.data(), inputs.size(), plugin);
        RreplenishLayerAndOutput(layer, "matmul_op_int8_dynamic", {output_name},
                                 test_mode);
      } else {
        VLOG(3) << "Convert a fluid matmul_op_int8_static to TensorRT "
                   "MatmulPluginLayer";
        plugin::MatmulPlugin* plugin = new plugin::MatmulPlugin(
            dims_x, dims_y, transpose_X, transpose_Y, alpha);
        std::vector<nvinfer1::ITensor*> inputs{input1, input2};
        layer = engine_->AddPluginV2IOExt(inputs.data(), inputs.size(), plugin);
        RreplenishLayerAndOutput(layer, "matmul_op_int8_static", {output_name},
                                 test_mode);
      }
    } else {
      VLOG(3) << "Convert a fluid matmul_op_float to TensorRT ";
      layer =
          TRT_ENGINE_ADD_LAYER(engine_, MatrixMultiply, *input1,
                               matrix_operation_X, *input2, matrix_operation_Y);
      if (alpha == 1) {
        RreplenishLayerAndOutput(layer, "matmul_op_float_no_alpha",
                                 {output_name}, test_mode);
      } else {
        layer->setName(
            ("matmul_op_float_has_alpha: MatrixMultiplyLayer (Output: " +
             output_name + ")")
                .c_str());
        // IScaleLayer requires the input must have at least
        // three dimensions in static shape mode and at least
        // four dimensions in dynamic shape mode.
        auto* matmul_out = layer->getOutput(0);
        nvinfer1::Dims out_shape = matmul_out->getDimensions();
        const int out_dims = out_shape.nbDims;
        bool need_change_dim = false;

        if (engine_->with_dynamic_shape()) {
          if (out_dims == 3) {
            need_change_dim = true;
          }
        } else {
          if (out_dims == 2) {
            need_change_dim = true;
          }
        }

        if (need_change_dim) {
          nvinfer1::Dims reshape_dim;
          reshape_dim.nbDims = out_dims + 1;
          reshape_dim.d[out_dims] = 1;
          for (int i = 0; i < out_dims; i++) {
            reshape_dim.d[i] = out_shape.d[i];
          }

          auto* reshape_layer =
              TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *matmul_out);
          reshape_layer->setReshapeDimensions(reshape_dim);
          matmul_out = reshape_layer->getOutput(0);
          reshape_layer->setName(("matmul_op_float_has_alpha_reshape_before: "
                                  "ShuffleLayer (Output: " +
                                  output_name + ")")
                                     .c_str());
        }

        auto create_weights = [&](float data,
                                  const std::string& type) -> float* {
          std::unique_ptr<framework::Tensor> tmp_tensor(
              new framework::Tensor());
          tmp_tensor->Resize({1});
          auto* tmp_data =
              tmp_tensor->mutable_data<float>(platform::CPUPlace());
          tmp_data[0] = data;
          engine_->SetWeights(output_name + "_add_scale_op_" + type,
                              std::move(tmp_tensor));
          return tmp_data;
        };
        float* alpha_data = create_weights(alpha, "alpha");
        float* shift_data = create_weights(0.0, "shift");
        float* power_data = create_weights(1.0, "power");
        TensorRTEngine::Weight nv_alpha{nvinfer1::DataType::kFLOAT,
                                        static_cast<void*>(alpha_data), 1};
        TensorRTEngine::Weight nv_shift{nvinfer1::DataType::kFLOAT,
                                        static_cast<void*>(shift_data), 1};
        TensorRTEngine::Weight nv_power{nvinfer1::DataType::kFLOAT,
                                        static_cast<void*>(power_data), 1};
        auto* scale_layer = TRT_ENGINE_ADD_LAYER(
            engine_, Scale, *matmul_out, nvinfer1::ScaleMode::kUNIFORM,
            nv_shift.get(), nv_alpha.get(), nv_power.get());
        auto* scale_out = scale_layer->getOutput(0);
        scale_layer->setName(
            ("matmul_op_float_has_alpha: ScaleLayer (Output: " + output_name +
             ")")
                .c_str());

        if (need_change_dim) {
          auto* reshape_layer =
              TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *scale_out);
          reshape_layer->setReshapeDimensions(out_shape);
          scale_out = reshape_layer->getOutput(0);
          reshape_layer->setName(("matmul_op_float_has_alpha_reshape_after: "
                                  "ShuffleLayer (Output: " +
                                  output_name + ")")
                                     .c_str());
        }
        engine_->SetITensor(output_name, scale_out);
        if (test_mode) {  // the test framework can not determine which is the
                          // output, so place the declaration inside.
          engine_->DeclareOutput(output_name);
        }
      }
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(matmul, MatMulOpConverter);
