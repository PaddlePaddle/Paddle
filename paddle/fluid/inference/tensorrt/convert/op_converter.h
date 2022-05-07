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
                 const framework::Scope& scope, TensorRTEngine* engine,
                 bool test_mode = false) {
    framework::OpDesc op_desc(op, nullptr);

    OpConverter* it{nullptr};

    if (op_desc.Type() == "mul") {
      PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1UL,
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
      // TODO(xingzhaolong): all mul, sub, div
      // static std::unordered_set<std::string> add_weight_op_set {"add", "mul",
      // "sub", "div"};
      static std::unordered_set<std::string> add_weight_op_set{"add", "mul"};
      PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1UL,
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
            add_weight_op_set.count(op_type), 0,
            platform::errors::Unimplemented("Unsupported elementwise type %s",
                                            op_type.c_str()));
        it = Registry<OpConverter>::Global().Lookup(
            "elementwise_" + op_type + "_weight");
        PADDLE_ENFORCE_NOT_NULL(
            it, platform::errors::Unimplemented(
                    "no OpConverter for optype [%s]", op_desc.Type()));
      } else {
        PADDLE_ENFORCE_GT(
            add_tensor_op_set.count(op_type), 0,
            platform::errors::Unimplemented("Unsupported elementwise type %s",
                                            op_type.c_str()));
        it = Registry<OpConverter>::Global().Lookup("elementwise_" + op_type +
                                                    "_tensor");
      }
      PADDLE_ENFORCE_NOT_NULL(
          it, platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                              op_desc.Type()));
    }

    if (op_desc.Type() == "depthwise_conv2d") {
      it = Registry<OpConverter>::Global().Lookup("conv2d");
      PADDLE_ENFORCE_NOT_NULL(
          it, platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                              op_desc.Type()));
    }
    if (op_desc.Type() == "depthwise_conv2d_transpose") {
      it = Registry<OpConverter>::Global().Lookup("conv2d_transpose");
      PADDLE_ENFORCE_NOT_NULL(
          it, platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                              op_desc.Type()));
    }
    if (op_desc.Type() == "transpose2") {
      it = Registry<OpConverter>::Global().Lookup("transpose");
      PADDLE_ENFORCE_NOT_NULL(
          it, platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                              op_desc.Type()));
    }
    if (op_desc.Type() == "flatten2") {
      it = Registry<OpConverter>::Global().Lookup("flatten");
      PADDLE_ENFORCE_NOT_NULL(
          it, platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                              op_desc.Type()));
    }
    // reshape2 == reshape
    if (op_desc.Type() == "reshape2") {
      it = Registry<OpConverter>::Global().Lookup("reshape");
      PADDLE_ENFORCE_NOT_NULL(
          it, platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                              op_desc.Type()));
    }
    if (!it) {
      it = Registry<OpConverter>::Global().Lookup(op_desc.Type());
    }
    PADDLE_ENFORCE_NOT_NULL(
        it, platform::errors::Unimplemented("no OpConverter for optype [%s]",
                                            op_desc.Type()));

    it->SetEngine(engine);
    VLOG(3) << "convert op: " << op_desc.Type();
    (*it)(op, scope, test_mode);

    VLOG(3) << "finish convert op: " << op_desc.Type();

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
        VLOG(3) << "name:  " << input_tensor_name << "; type: ";
        switch(input_itensor->getType())
            {
            case nvinfer1::DataType::kFLOAT: VLOG(3) << "kFLOAT"; break;
            case nvinfer1::DataType::kHALF: VLOG(3) << "kHALF"; break;
            default: break;
            }
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
                    const framework::Scope& scope, TensorRTEngine* engine) {
    std::unique_lock<std::mutex> lk(mut_);
    for (int i = 0; i < block.ops_size(); i++) {
      const auto& op = block.ops(i);

//      framework::OpDesc op_desc_tmp(op, nullptr);
//      LOG(INFO) << op_desc_tmp.Type();
//      auto ins = op_desc_tmp.InputArgumentNames();
//      for (size_t k = 0; k < ins.size(); ++k) {
//        LOG(INFO) << ins[k];
//      }

      ConvertOp(op, parameters, scope, engine);
    }
  }

  // The scope  here should be inited with the parameter vars.
  void ConvertBlockToTRTEngine(
      framework::BlockDesc* block_desc, const framework::Scope& scope,
      const std::vector<std::string>& inputs,
      const std::unordered_set<std::string>& parameters,
      const std::vector<std::string>& outputs, TensorRTEngine* engine) {
    engine->InitNetwork();
    bool all_dynamic_shape_set = true;
    for (auto& input : inputs) {
      if (parameters.count(input)) continue;
      auto* var = block_desc->FindVar(input);
      PADDLE_ENFORCE_NOT_NULL(
          var, platform::errors::NotFound("no variable called %s in block.",
                                          input.c_str()));
      PADDLE_ENFORCE_EQ(
          var->GetType(), FluidDT::VarType_Type_LOD_TENSOR,
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
            PADDLE_ENFORCE_EQ(min_input_shape[i], optim_input_shape[i],
                              platform::errors::InvalidArgument(
                                  "The dim (%d) of the min_input_shape and "
                                  "optim_input_shape should be same."));
          }
        }

        nvinfer1::DataType dtype = FluidDataType2TRT(var->Proto()->type().lod_tensor().tensor().data_type());
        if(input.rfind("fill_constant_batch_size", 0) == 0) {
            dtype = nvinfer1::DataType::kHALF;
        }
        engine->DeclareInput(
            input, dtype,
            Vec2TRT_Dims(input_shape, input, true));
        
        VLOG(3) << "name:  " << input << "; type: ";
        switch(dtype)
            {
            case nvinfer1::DataType::kFLOAT: VLOG(3) << "kFLOAT"; break;
            case nvinfer1::DataType::kHALF: VLOG(3) << "kHALF"; break;
            default: break;
            }

#endif
      } else {
        engine->DeclareInput(
            input, FluidDataType2TRT(
                       var->Proto()->type().lod_tensor().tensor().data_type()),
            Vec2TRT_Dims(var_shape, input));
      }
    }
    PADDLE_ENFORCE_EQ(all_dynamic_shape_set, true,
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

  void RreplenishLayerAndOutput(
      nvinfer1::ILayer* layer, const std::string& layer_type,
      const std::vector<std::string>& output_tensor_names,
      bool test_mode = false) {
    size_t num_out = output_tensor_names.size();
    for (size_t i = 0; i < num_out; i++) {
      layer->getOutput(i)->setName(output_tensor_names[i].c_str());
      engine_->SetITensor(output_tensor_names[i], layer->getOutput(i));
      if (test_mode) {
        engine_->DeclareOutput(output_tensor_names[i]);
      }
    }
    layer->setName(
        (layer_type + " (Output: " + output_tensor_names[0] + ")").c_str());
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
