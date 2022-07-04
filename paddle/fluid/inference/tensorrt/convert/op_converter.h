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

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/utils/singleton.h"

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * Convert Op from Fluid to TensorRT Engine.
 */
class OpConverter {
 public:
  OpConverter() {}

  // Converter logic for an op.
  virtual void operator()(const framework::proto::OpDesc& op,
                          const framework::Scope& scope,
                          bool test_mode = false) {}

  // Convert a single fluid operator and add the corresponding layer to TRT.
  // test_mode: whether the instance executes in an unit test.
  void ConvertOp(const framework::proto::OpDesc& op,
                 const std::unordered_set<std::string>& parameters,
                 const framework::Scope& scope,
                 TensorRTEngine* engine,
                 bool test_mode = false) {
    framework::OpDesc op_desc(op, nullptr);

    OpConverter* it{nullptr};

    if (op_desc.Type() == "mul") {
      PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
                        1UL,
                        platform::errors::InvalidArgument(
                            "The input op mul's Input(\"Y\")."
                            "size() should equal to 1, but reveceid "
                            "Input(\"Y\").size() = %u.",
                            op_desc.Input("Y").size()));
      std::string Y = op_desc.Input("Y")[0];
      if (parameters.count(Y)) {
        it = Registry<OpConverter>::Global().Lookup("fc");
      }
    }
    if (op_desc.Type().find("elementwise") != std::string::npos) {
      static std::unordered_set<std::string> add_tensor_op_set{
          "add", "mul", "sub", "div", "max", "min", "pow"};
      static std::unordered_set<std::string> add_weight_op_set{
          "add", "mul", "sub", "div", "pow"};
      PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(),
                        1UL,
                        platform::errors::InvalidArgument(
                            "The input op's Input(\"Y\")."
                            "size() should equal to 1, but reveceid "
                            "Input(\"Y\").size() = %u.",
                            op_desc.Input("Y").size()));
      int op_type_len = op_desc.Type().size();
      std::string op_type = op_desc.Type().substr(op_type_len - 3, op_type_len);
      std::string Y = op_desc.Input("Y")[0];
      if (parameters.count(Y)) {
        PADDLE_ENFORCE_GT(
            add_weight_op_set.count(op_type),
            0,
            platform::errors::Unimplemented("Unsupported elementwise type %s",
                                            op_type.c_str()));
        it = Registry<OpConverter>::Global().Lookup("elementwise_" + op_type +
                                                    "_weight");
        PADDLE_ENFORCE_NOT_NULL(
            it,
            platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                            op_desc.Type()));
      } else {
        PADDLE_ENFORCE_GT(
            add_tensor_op_set.count(op_type),
            0,
            platform::errors::Unimplemented("Unsupported elementwise type %s",
                                            op_type.c_str()));
        it = Registry<OpConverter>::Global().Lookup("elementwise_" + op_type +
                                                    "_tensor");
      }
      PADDLE_ENFORCE_NOT_NULL(
          it,
          platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                          op_desc.Type()));
    }

    if (op_desc.Type() == "depthwise_conv2d") {
      it = Registry<OpConverter>::Global().Lookup("conv2d");
      PADDLE_ENFORCE_NOT_NULL(
          it,
          platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                          op_desc.Type()));
    }
    if (op_desc.Type() == "depthwise_conv2d_transpose") {
      it = Registry<OpConverter>::Global().Lookup("conv2d_transpose");
      PADDLE_ENFORCE_NOT_NULL(
          it,
          platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                          op_desc.Type()));
    }
    if (op_desc.Type() == "transpose2") {
      it = Registry<OpConverter>::Global().Lookup("transpose");
      PADDLE_ENFORCE_NOT_NULL(
          it,
          platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                          op_desc.Type()));
    }
    if (op_desc.Type() == "flatten2") {
      it = Registry<OpConverter>::Global().Lookup("flatten");
      PADDLE_ENFORCE_NOT_NULL(
          it,
          platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                          op_desc.Type()));
    }
    // reshape2 == reshape
    if (op_desc.Type() == "reshape2") {
      it = Registry<OpConverter>::Global().Lookup("reshape");
      PADDLE_ENFORCE_NOT_NULL(
          it,
          platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                          op_desc.Type()));
    }
    if (!it) {
      it = Registry<OpConverter>::Global().Lookup(op_desc.Type());
    }
    PADDLE_ENFORCE_NOT_NULL(
        it,
        platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                        op_desc.Type()));

    it->SetEngine(engine);
    (*it)(op, scope, test_mode);

    size_t output_num = op_desc.OutputNames().size();
    // only one out settensordynamicRange
    if (op_desc.HasAttr("out_threshold")) {
      float out_scale =
          BOOST_GET_CONST(float, op_desc.GetAttr("out_threshold"));
      std::string output_name = "";
      if (op_desc.HasOutput("Output")) {
        output_name = op_desc.Output("Output").front();
      } else if (op_desc.HasOutput("Out")) {
        output_name = op_desc.Output("Out").front();
      } else if (op_desc.HasOutput("Y")) {
        output_name = op_desc.Output("Y").front();
      } else {
        PADDLE_THROW(
            platform::errors::NotFound("Op %s has out threshold but doesn't "
                                       "have an output named \"Output\", "
                                       "\"Out\" or \"Y\".",
                                       op_desc.Type()));
      }
      auto* output_itensor = engine->GetITensor(output_name);
      engine->SetTensorDynamicRange(output_itensor, out_scale);
      VLOG(1) << "Set out scale = " << out_scale << " for tensor "
              << output_name << ".";
    }
    // outs settensordynamicRange
    for (size_t i = 0; i < output_num; ++i) {
      if (op_desc.HasAttr("out_" + std::to_string(i) + "_threshold")) {
        float out_scale = BOOST_GET_CONST(
            float, op_desc.GetAttr("out_" + std::to_string(i) + "_threshold"));
        std::string output_name =
            op_desc.Output(op_desc.OutputNames()[i]).front();
        auto* output_itensor = engine->GetITensor(output_name);
        engine->SetTensorDynamicRange(output_itensor, out_scale);
        VLOG(1) << "Set out scale = " << out_scale << " for tensor "
                << output_name << ".";
      }
    }

    // quant_dequant_linear support for paddle trt

    std::vector<std::string> inputs_name = op_desc.InputNames();
    std::vector<std::string> outputs_name = op_desc.OutputNames();

    for (size_t i = 0; i < inputs_name.size(); i++) {
      if (op_desc.HasAttr(inputs_name[i])) {
        std::string input_tensor_name = op_desc.Input(inputs_name[i])[0];
        auto* input_itensor = engine->GetITensor(input_tensor_name);
        float input_scale =
            BOOST_GET_CONST(float, op_desc.GetAttr(inputs_name[i]));
        engine->SetTensorDynamicRange(input_itensor, input_scale);
        VLOG(1) << "Set input tensor scale = " << input_scale
                << " for tensor: " << input_tensor_name << ".";
      }
    }
    for (size_t i = 0; i < outputs_name.size(); i++) {
      if (op_desc.HasAttr(outputs_name[i])) {
        std::string output_tensor_name = op_desc.Output(outputs_name[i])[0];
        auto* output_itensor = engine->GetITensor(output_tensor_name);
        float output_scale =
            BOOST_GET_CONST(float, op_desc.GetAttr(outputs_name[i]));
        engine->SetTensorDynamicRange(output_itensor, output_scale);
        VLOG(1) << "Set output tensor scale = " << output_scale
                << " for tensor: " << output_tensor_name << ".";
      }
    }
  }

  // Convert a fluid block to tensorrt network, NOTE it just convert operators,
  // the INetwork's inputs and outputs should specified in some other modules.
  void ConvertBlock(const framework::proto::BlockDesc& block,
                    const std::unordered_set<std::string>& parameters,
                    const framework::Scope& scope,
                    TensorRTEngine* engine) {
    std::unique_lock<std::mutex> lk(mut_);
    for (int i = 0; i < block.ops_size(); i++) {
      const auto& op = block.ops(i);
      ConvertOp(op, parameters, scope, engine);
    }
  }

  // The scope  here should be inited with the parameter vars.
  void ConvertBlockToTRTEngine(
      framework::BlockDesc* block_desc,
      const framework::Scope& scope,
      const std::vector<std::string>& inputs,
      const std::unordered_set<std::string>& parameters,
      const std::vector<std::string>& outputs,
      TensorRTEngine* engine) {
    engine->InitNetwork();
    bool all_dynamic_shape_set = true;
    for (auto& input : inputs) {
      if (parameters.count(input)) continue;
      auto* var = block_desc->FindVar(input);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          platform::errors::NotFound("no variable called %s in block.",
                                     input.c_str()));
      PADDLE_ENFORCE_EQ(
          var->GetType(),
          FluidDT::VarType_Type_LOD_TENSOR,
          platform::errors::InvalidArgument("TensorRT engine only takes "
                                            "LoDTensor as input"));
      auto var_shape = var->GetShape();
      if (engine->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
        auto min_input_shape = engine->min_input_shape()[input];
        auto max_input_shape = engine->max_input_shape()[input];
        auto optim_input_shape = engine->optim_input_shape()[input];
        size_t ranks = min_input_shape.size();
        if (ranks == 0) {
          all_dynamic_shape_set = false;
          LOG(INFO) << "trt input [" << input.c_str()
                    << "] dynamic shape info not set, please check and retry.";
          // check other input
          continue;
        }
        std::vector<int64_t> input_shape;
        input_shape.push_back(-1);
        for (size_t i = 1; i < ranks; i++) {
          if (min_input_shape[i] != max_input_shape[i]) {
            input_shape.push_back(-1);
          } else {
            input_shape.push_back(min_input_shape[i]);
            // the i dimension should be same.
            PADDLE_ENFORCE_EQ(min_input_shape[i],
                              optim_input_shape[i],
                              platform::errors::InvalidArgument(
                                  "The dim (%d) of the min_input_shape and "
                                  "optim_input_shape should be same."));
          }
        }
        engine->DeclareInput(
            input,
            FluidDataType2TRT(
                var->Proto()->type().lod_tensor().tensor().data_type()),
            Vec2TRT_Dims(input_shape, input, true));
#endif
      } else {
        engine->DeclareInput(
            input,
            FluidDataType2TRT(
                var->Proto()->type().lod_tensor().tensor().data_type()),
            Vec2TRT_Dims(var_shape, input));
        VLOG(1) << "Set trt input [ " << input << "] type is "
                << var->Proto()->type().lod_tensor().tensor().data_type();
      }
    }
    PADDLE_ENFORCE_EQ(all_dynamic_shape_set,
                      true,
                      platform::errors::InvalidArgument(
                          "some trt inputs dynamic shape info not set, "
                          "check the INFO log above for more details."));
    framework::proto::BlockDesc* block_proto = block_desc->Proto();
    ConvertBlock(*block_proto, parameters, scope, engine);
    for (auto& output : outputs) {
      engine->DeclareOutput(output);
    }
    engine->FreezeNetwork();
    engine->ClearWeights();
  }

  // rank(result) = rank(input)
  nvinfer1::ITensor* Gather(nvinfer1::ITensor* input,
                            const std::vector<int32_t> indices,
                            int axis = 0) {
    auto* indices_tensor = Add1DConstantLayer(indices, " ");
    auto* result =
        TRT_ENGINE_ADD_LAYER(engine_, Gather, *input, *indices_tensor, axis)
            ->getOutput(0);
    return result;
  }

  // paddle allows negative index
  // for axis length = 5, paddle allows [-5, 4]
  nvinfer1::ITensor* FixNegIndices(nvinfer1::ITensor* input_shape,
                                   nvinfer1::ITensor* indices) {
    int rank = input_shape->getDimensions().nbDims;
    std::vector<int32_t> zero = std::vector<int32_t>(rank, 0);
    std::vector<int32_t> minus_one = std::vector<int32_t>(rank, -1);
    nvinfer1::ITensor* zero_tensor = Add1DConstantLayer(zero);
    nvinfer1::ITensor* minus_one_tensor = Add1DConstantLayer(minus_one);
    // -1, 0
    auto* sign = Max(Min(indices, zero_tensor), minus_one_tensor);
    return Sub(indices, Prod(sign, input_shape));
  }

  nvinfer1::ITensor* Shape(nvinfer1::ITensor* input) {
    return TRT_ENGINE_ADD_LAYER(engine_, Shape, *input)->getOutput(0);
  }

  // Concat not make rank changed
  nvinfer1::ITensor* Concat(const std::vector<nvinfer1::ITensor*>& inputs,
                            int axis = 0) {
    auto* layer = TRT_ENGINE_ADD_LAYER(
        engine_, Concatenation, inputs.data(), inputs.size());
    if (axis != 0) layer->setAxis(axis);
    nvinfer1::ITensor* c = layer->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Sum(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kSUM)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Prod(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kPROD)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Min(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kMIN)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Max(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kMAX)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Sub(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kSUB)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Div(nvinfer1::ITensor* a, nvinfer1::ITensor* b) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(
            engine_, ElementWise, *a, *b, nvinfer1::ElementWiseOperation::kDIV)
            ->getOutput(0);
    return c;
  }

  nvinfer1::ITensor* Act(nvinfer1::ITensor* a,
                         nvinfer1::ActivationType act_type) {
    nvinfer1::ITensor* c =
        TRT_ENGINE_ADD_LAYER(engine_, Activation, *a, act_type)->getOutput(0);
    return c;
  }

  // Get element tensor of 1D shape tensor
  nvinfer1::ITensor* GetEleTensorOfShape(nvinfer1::ITensor* shape_tensor,
                                         int index,
                                         bool is_scalar = false) {
    auto* tensor =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Gather,
                             *shape_tensor,
                             *Add1DConstantLayer(index, " ", is_scalar),
                             0)
            ->getOutput(0);
    return tensor;
  }

  // Create and add Multi-D constant float layer
  nvinfer1::ITensor* AddConstantLayer(const float* data,
                                      const std::vector<int32_t>& weight_dims,
                                      const std::string& weight_name) {
    std::unique_ptr<framework::Tensor> tmp_tensor(new framework::Tensor());
    int data_size = std::accumulate(
        weight_dims.begin(), weight_dims.end(), 1, std::multiplies<int>());
    tmp_tensor->Resize({data_size});
    auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
    for (int i = 0; i < data_size; i++) {
      tmp_data[i] = data[i];
    }
    engine_->SetWeights(weight_name, std::move(tmp_tensor));

    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(tmp_data),
                                  static_cast<size_t>(data_size)};
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = weight_dims.size();
    for (size_t i = 0; i < weight_dims.size(); i++)
      trt_dims.d[i] = weight_dims[i];
    auto const_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, trt_dims, weight.get());
    return const_layer->getOutput(0);
  }

  // Create and add 1D constant float layer
  nvinfer1::ITensor* Add1DConstantLayer(const std::vector<float>& data,
                                        const std::string& weight_name = "",
                                        bool scalar = false) {
    std::unique_ptr<framework::Tensor> tmp_tensor(new framework::Tensor());
    int data_size = data.size();
    tmp_tensor->Resize({data_size});
    auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
    for (int i = 0; i < data_size; i++) {
      tmp_data[i] = data[i];
    }
    engine_->SetWeights(weight_name, std::move(tmp_tensor));

    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(tmp_data),
                                  static_cast<size_t>(data_size)};
    nvinfer1::Dims input_shape;
    input_shape.nbDims = scalar ? 0 : 1;
    input_shape.d[0] = data_size;
    auto const_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, input_shape, weight.get());
    return const_layer->getOutput(0);
  }

  // Create and add 1D constant layer
  nvinfer1::ITensor* Add1DConstantLayer(const std::vector<int>& data,
                                        const std::string& weight_name = "",
                                        bool scalar = false) {
    std::unique_ptr<framework::Tensor> tmp_tensor(new framework::Tensor());
    int data_size = data.size();
    tmp_tensor->Resize({data_size});
    auto* tmp_data = tmp_tensor->mutable_data<int>(platform::CPUPlace());
    for (int i = 0; i < data_size; i++) {
      tmp_data[i] = data[i];
    }
    engine_->SetWeights(weight_name, std::move(tmp_tensor));

    TensorRTEngine::Weight weight{nvinfer1::DataType::kINT32,
                                  static_cast<void*>(tmp_data),
                                  static_cast<size_t>(data_size)};
    nvinfer1::Dims input_shape;
    input_shape.nbDims = scalar ? 0 : 1;
    input_shape.d[0] = data_size;
    auto const_layer =
        TRT_ENGINE_ADD_LAYER(engine_, Constant, input_shape, weight.get());
    return const_layer->getOutput(0);
  }

  nvinfer1::ITensor* Add1DConstantLayer(nvinfer1::Dims data,
                                        const std::string& weight_name = "",
                                        bool scalar = false) {
    std::vector<int> tmp_data;
    for (int i = 0; i < data.nbDims; i++) tmp_data.push_back(data.d[i]);
    return Add1DConstantLayer(tmp_data, weight_name, scalar);
  }

  nvinfer1::ITensor* Add1DConstantLayer(int32_t data,
                                        const std::string& weight_name = "",
                                        bool scalar = false) {
    std::vector<int> tmp_data;
    tmp_data.push_back(data);
    return Add1DConstantLayer(tmp_data, weight_name, scalar);
  }

  void RreplenishLayerAndOutput(
      nvinfer1::ILayer* layer,
      const std::string& layer_type,
      const std::vector<std::string>& output_tensor_names,
      bool test_mode = false) {
    size_t num_out = output_tensor_names.size();
    std::string layer_name = layer_type + " (Output: ";
    for (size_t i = 0; i < num_out; i++) {
      layer->getOutput(i)->setName(output_tensor_names[i].c_str());
      engine_->SetITensor(output_tensor_names[i], layer->getOutput(i));
      if (test_mode) {
        engine_->DeclareOutput(output_tensor_names[i]);
      }
      layer_name += output_tensor_names[i];
      if (i != num_out - 1) layer_name += ", ";
    }
    layer->setName((layer_name + ")").c_str());
  }
  void SetEngine(TensorRTEngine* engine) { engine_ = engine; }

  virtual ~OpConverter() {}

  // TensorRT engine
  TensorRTEngine* engine_{nullptr};

 protected:
  bool test_mode_;

 private:
  // registered op converter map, whose key is the fluid op type, and value is
  // the pointer position of corresponding OpConverter class.
  std::unordered_map<std::string, OpConverter*> converters_;
  // fluid inference scope
  framework::Scope* scope_{nullptr};
  std::mutex mut_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

#define REGISTER_TRT_OP_CONVERTER(op_type__, Converter__)                      \
  struct trt_##op_type__##_converter : public ::paddle::framework::Registrar { \
    trt_##op_type__##_converter() {                                            \
      ::paddle::inference::Registry<                                           \
          paddle::inference::tensorrt::OpConverter>::Global()                  \
          .Register<::paddle::inference::tensorrt::Converter__>(#op_type__);   \
    }                                                                          \
  };                                                                           \
  trt_##op_type__##_converter trt_##op_type__##_converter__;                   \
  int TouchConverterRegister_##op_type__() {                                   \
    trt_##op_type__##_converter__.Touch();                                     \
    return 0;                                                                  \
  }

#define USE_TRT_CONVERTER(op_type__)                   \
  extern int TouchConverterRegister_##op_type__();     \
  static int use_op_converter_trt_##op_type__ UNUSED = \
      TouchConverterRegister_##op_type__();
