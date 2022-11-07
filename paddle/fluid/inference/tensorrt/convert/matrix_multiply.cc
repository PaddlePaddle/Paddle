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
#include "paddle/fluid/inference/tensorrt/plugin/matrix_multiply_op_int8_plugin.h"

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
 * MatrixMultiplyOp, IMatrixMultiplyLayer in TRT. This Layer doesn't has weights.
 */
class MatrixMultiplyOpConverter : public OpConverter {
 public:
   nvinfer1::ILayer* reshape_before_fc(nvinfer1::ITensor* before_fc,
                                      nvinfer1::Dims tensor_dim,
                                      int col_dims,
                                      std::string output_name) {
    // add shuffle before fc
    nvinfer1::Dims reshape_before_fc_dim;
    reshape_before_fc_dim.nbDims = col_dims+1;
    // reshape "* x k "

    nvinfer1::ITensor* filal_reshape_before_fc_shape_tensor = nullptr;

    if (!engine_->with_dynamic_shape()) {
      for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
        reshape_before_fc_dim.d[i] = 1;
      }
      for (int i = 0; i < tensor_dim.nbDims; i++) {
        if (i < col_dims) {
          reshape_before_fc_dim.d[i] = 0;
        } else {
          reshape_before_fc_dim.d[col_dims] *= tensor_dim.d[i];
        }
      }
    } else {
      std::vector<nvinfer1::ITensor*> reshape_before_fc_shape_tensor;
      nvinfer1::ITensor* input_shape_tensor = Shape(before_fc);

      for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
        reshape_before_fc_shape_tensor.push_back(Add1DConstantLayer(1));
      }
      for (int i = 0; i < tensor_dim.nbDims; i++) {
        if (i < col_dims) {
          reshape_before_fc_shape_tensor[i] =
              GetEleTensorOfShape(input_shape_tensor, i);
        } else {
          reshape_before_fc_shape_tensor[col_dims] =
              Prod(GetEleTensorOfShape(input_shape_tensor, i),
                   reshape_before_fc_shape_tensor[col_dims]);
        }
      }
      filal_reshape_before_fc_shape_tensor =
          Concat(reshape_before_fc_shape_tensor);
    }

    auto* reshape_before_fc_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *before_fc);
    if (!engine_->with_dynamic_shape()) {
      reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
    } else {
      reshape_before_fc_layer->setInput(1,
                                        *filal_reshape_before_fc_shape_tensor);
    }
    reshape_before_fc_layer->setName(
        ("fc_op_reshape_before_fc: Shuffle (Output: " + output_name + ")")
            .c_str());
    return reshape_before_fc_layer;
  }

  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid matrix_multiply op to tensorrt MatrixMultiply layer ";
    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;

    int32_t x_num_col_dims = PADDLE_GET_CONST(int32_t, op_desc.GetAttr("x_num_col_dims"));
    bool transpose_X = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_X"));
    bool transpose_Y = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));
    float alpha = PADDLE_GET_CONST(float, op_desc.GetAttr("alpha"));
    bool y_is_weight = PADDLE_GET_CONST(bool, op_desc.GetAttr("y_is_weight"));
        


    nvinfer1::MatrixOperation matrix_operation_X =
        transpose_X ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;
    nvinfer1::MatrixOperation matrix_operation_Y =
        transpose_Y ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;

    // Declare inputs
    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    nvinfer1::ITensor* Y = nullptr;
    if(y_is_weight){
      Y=ConvertWeight2ITensor(scope,op_desc.Input("Y").front());

    }else{
      Y = engine_->GetITensor(op_desc.Input("Y").front());
    }
    auto output_name = op_desc.Output("Out").front();
    if (!engine_->with_dynamic_shape()) {
      x_num_col_dims--;
    }
    nvinfer1::Dims dims_x = X->getDimensions();
    nvinfer1::Dims dims_y = Y->getDimensions();

    auto* reshape_before_fc_layer =
        reshape_before_fc(X, dims_x, x_num_col_dims, output_name);
    auto* reshape_itensor = reshape_before_fc_layer->getOutput(0);

    if (op_desc.HasAttr("Input_scale") || op_desc.HasAttr("X")) {
      float in_scale = 0;
      if (op_desc.HasAttr("Input_scale")) {
        in_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("Input_scale"));
      } else {
        in_scale = PADDLE_GET_CONST(float, op_desc.GetAttr("X"));
      }
      engine_->SetTensorDynamicRange(X, in_scale);
      engine_->SetTensorDynamicRange(reshape_itensor, in_scale);
    }
    if (op_desc.HasAttr("support_int8") &&
        PADDLE_GET_CONST(bool, op_desc.GetAttr("support_int8")) &&
        engine_->precision() == AnalysisConfig::Precision::kInt8 &&
        platform::GetGPUComputeCapability(0) >= 75) {
      if (engine_->with_dynamic_shape()) {
        VLOG(3) << "Convert a fluid matrix_multiply_op_int8_dynamic to TensorRT "
                   "MatmulPluginLayer";
        plugin::MatmulPluginDynamic* plugin =
            new plugin::MatmulPluginDynamic(transpose_X, transpose_Y, alpha);
        std::vector<nvinfer1::ITensor*> inputs{X, Y};
        layer = engine_->AddDynamicPlugin(inputs.data(), inputs.size(), plugin);
        RreplenishLayerAndOutput(
            layer, "matrix_multiply_op_int8_dynamic", {output_name}, test_mode);
      } else {
        VLOG(3) << "Convert a fluid matrix_multiply_op_int8_static to TensorRT "
                   "MatmulPluginLayer";
        plugin::MatmulPlugin* plugin = new plugin::MatmulPlugin(
            reshape_itensor->getDimensions(), dims_y, transpose_X, transpose_Y, alpha);
        std::vector<nvinfer1::ITensor*> inputs{X, Y};
        layer = engine_->AddPluginV2IOExt(inputs.data(), inputs.size(), plugin);
        RreplenishLayerAndOutput(
            layer, "matrix_multiply_op_int8_static", {output_name}, test_mode);
      }
    } else {
      VLOG(3) << "Convert a fluid matrix_multiply_op_float to TensorRT ";
      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   MatrixMultiply,
                                   *X,
                                   matrix_operation_X,
                                   *Y,
                                   matrix_operation_Y);
      auto* matrix_multiply_out = layer->getOutput(0);
      if(op_desc.HasAttr("out_threshold")){
        float out_threshold = PADDLE_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        engine_->SetTensorDynamicRange(matrix_multiply_out, out_threshold);
      }                      
      if (alpha == 1) {
        RreplenishLayerAndOutput(
            layer, "matrix_multiply_op_float_no_alpha", {output_name}, test_mode);
      } else {
        layer->setName(
            ("matrix_multiply_op_float_has_alpha: MatrixMultiplyLayer (Output: " +
             output_name + ")")
                .c_str());
        // IScaleLayer requires the input must have at least
        // three dimensions in static shape mode and at least
        // four dimensions in dynamic shape mode.

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
        TensorRTEngine::Weight nv_alpha{
            nvinfer1::DataType::kFLOAT, static_cast<void*>(alpha_data), 1};
        TensorRTEngine::Weight nv_shift{
            nvinfer1::DataType::kFLOAT, static_cast<void*>(shift_data), 1};
        TensorRTEngine::Weight nv_power{
            nvinfer1::DataType::kFLOAT, static_cast<void*>(power_data), 1};
        auto* scale_layer = TRT_ENGINE_ADD_LAYER(engine_,
                                                 Scale,
                                                 *matrix_multiply_out,
                                                 nvinfer1::ScaleMode::kUNIFORM,
                                                 nv_shift.get(),
                                                 nv_alpha.get(),
                                                 nv_power.get());
        auto* scale_out = scale_layer->getOutput(0);
        scale_layer->setName(
            ("matrix_multiply_op_float_has_alpha: ScaleLayer (Output: " + output_name +
             ")")
                .c_str());
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

REGISTER_TRT_OP_CONVERTER(matrix_multiply, MatrixMultiplyOpConverter);
