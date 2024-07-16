// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/new_executor/instruction/tensorrt_engine_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/platform/tensorrt/engine_params.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"

namespace paddle {
namespace framework {

using TensorRTEngine = paddle::platform::TensorRTEngine;

TensorRTEngineInstruction::TensorRTEngineInstruction(
    size_t id,
    const platform::Place &place,
    ::pir::Operation *op,
    const ValueExecutionInfo *value_exec_info)
    : InstructionBase(id, place), value_exec_info_(value_exec_info) {
  auto op_attributes = op->attributes();

  VLOG(6) << "Start Build engine";
  auto engine_serialized_data = op_attributes.at("engine_serialized_data")
                                    .dyn_cast<pir::StrAttribute>()
                                    .AsString();
  workspace_size_ =
      op_attributes.at("workspace_size").dyn_cast<pir::Int64Attribute>().data();
  allow_build_at_runtime_ = op_attributes.at("allow_build_at_runtime")
                                .dyn_cast<pir::BoolAttribute>()
                                .data();
  auto output_names_attrs = op_attributes.at("output_names")
                                .dyn_cast<pir::ArrayAttribute>()
                                .AsVector();
  for (auto output_names_attr : output_names_attrs) {
    output_names_.push_back(
        output_names_attr.dyn_cast<pir::StrAttribute>().AsString());
  }
  auto outputs_rank_attrs = op_attributes.at("outputs_rank")
                                .dyn_cast<pir::ArrayAttribute>()
                                .AsVector();
  for (auto outputs_rank_attr : outputs_rank_attrs) {
    outputs_rank_.push_back(
        outputs_rank_attr.dyn_cast<pir::Int32Attribute>().data());
  }
  auto outputs_dtype_attrs = op_attributes.at("outputs_dtype")
                                 .dyn_cast<pir::ArrayAttribute>()
                                 .AsVector();
  for (auto outputs_dtype_attr : outputs_dtype_attrs) {
    outputs_dtype_.push_back(
        outputs_dtype_attr.dyn_cast<paddle::dialect::DataTypeAttribute>()
            .data());
  }

  op_ = op;
  auto input_names_attrs = op_attributes.at("input_names")
                               .dyn_cast<pir::ArrayAttribute>()
                               .AsVector();
  for (auto input_names_attr : input_names_attrs) {
    input_names_.push_back(
        input_names_attr.dyn_cast<pir::StrAttribute>().AsString());
  }

  std::vector<std::string> dynamic_shape_names;
  std::vector<int> dynamic_shape_lens;
  std::vector<int> min_input_shape_vector;
  std::vector<int> max_input_shape_vector;
  std::vector<int> opt_input_shape_vector;

  auto dynamic_shape_names_attrs = op_attributes.at("dynamic_shape_names")
                                       .dyn_cast<pir::ArrayAttribute>()
                                       .AsVector();
  auto min_input_shapes_attrs = op_attributes.at("min_input_shape_vector")
                                    .dyn_cast<pir::ArrayAttribute>()
                                    .AsVector();
  auto max_input_shapes_attrs = op_attributes.at("max_input_shape_vector")
                                    .dyn_cast<pir::ArrayAttribute>()
                                    .AsVector();
  auto opt_input_shapes_attrs = op_attributes.at("opt_input_shape_vector")
                                    .dyn_cast<pir::ArrayAttribute>()
                                    .AsVector();
  auto dynamic_shape_lens_attrs = op_attributes.at("dynamic_shape_lens")
                                      .dyn_cast<pir::ArrayAttribute>()
                                      .AsVector();
  for (auto dynamic_shape_names_attr : dynamic_shape_names_attrs) {
    dynamic_shape_names.push_back(
        dynamic_shape_names_attr.dyn_cast<pir::StrAttribute>().AsString());
  }
  for (auto dynamic_shape_lens_attr : dynamic_shape_lens_attrs) {
    dynamic_shape_lens.push_back(
        dynamic_shape_lens_attr.dyn_cast<pir::Int32Attribute>().data());
  }
  for (auto min_input_shapes_attr : min_input_shapes_attrs) {
    min_input_shape_vector.push_back(
        min_input_shapes_attr.dyn_cast<pir::Int32Attribute>().data());
  }
  for (auto max_input_shapes_attr : max_input_shapes_attrs) {
    max_input_shape_vector.push_back(
        max_input_shapes_attr.dyn_cast<pir::Int32Attribute>().data());
  }
  for (auto opt_input_shapes_attr : opt_input_shapes_attrs) {
    opt_input_shape_vector.push_back(
        opt_input_shapes_attr.dyn_cast<pir::Int32Attribute>().data());
  }

  int idx = 0;
  std::vector<std::vector<int>> min_input_shapes;
  std::vector<std::vector<int>> max_input_shapes;
  std::vector<std::vector<int>> opt_input_shapes;
  for (size_t i = 0; i < dynamic_shape_lens.size(); ++i) {
    std::vector<int> tmp1, tmp2, tmp3;
    for (int j = 0; j < dynamic_shape_lens[i]; ++j) {
      tmp1.push_back(min_input_shape_vector[idx]);
      tmp2.push_back(max_input_shape_vector[idx]);
      tmp3.push_back(opt_input_shape_vector[idx++]);
    }
    min_input_shapes.emplace_back(tmp1);
    max_input_shapes.emplace_back(tmp2);
    opt_input_shapes.emplace_back(tmp3);
  }

  paddle::platform::EngineParams params;
  params.max_workspace_size = workspace_size_;
  params.device_id = place.device;

  for (size_t i = 0; i < dynamic_shape_names.size(); ++i) {
    params.min_input_shape.insert(
        std::make_pair(dynamic_shape_names[i], min_input_shapes[i]));
    params.max_input_shape.insert(
        std::make_pair(dynamic_shape_names[i], max_input_shapes[i]));
    params.optim_input_shape.insert(
        std::make_pair(dynamic_shape_names[i], opt_input_shapes[i]));
  }

  auto converter_debug_info = op_attributes.at("converter_debug_info")
                                  .dyn_cast<pir::StrAttribute>()
                                  .AsString();
  VLOG(6) << "======== TensorRT Graph Converter Info in tensorrt_engine_op("
          << op_ << "):=======";
  VLOG(6) << converter_debug_info;
  VLOG(6) << "================================================================="
             "===============";
  trt_engine_ = std::make_unique<paddle::platform::TensorRTEngine>(
      params, paddle::platform::NaiveLogger::Global());
  trt_engine_->Deserialize(engine_serialized_data);

  VLOG(6) << "Finish build engine for: " << op_name_;

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(place),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  InitInputsOutputsIds(op, *value_exec_info_);
  VLOG(6) << "finish process inputs outputs index";
}

static void RuntimeDynamicShapeCheck(
    const std::string &x,
    const std::vector<int32_t> &runtime_input_shape,
    const std::vector<int32_t> &min_input_shape,
    const std::vector<int32_t> &max_input_shape) {
  auto is_input_shape_valid =
      [&](const std::vector<int32_t> &runtime_input_shape,
          const std::vector<int32_t> &min_input_shape,
          const std::vector<int32_t> &max_input_shape) -> bool {
    for (size_t i = 0; i < runtime_input_shape.size(); i++) {
      if (runtime_input_shape[i] <= max_input_shape[i] &&
          runtime_input_shape[i] >= min_input_shape[i]) {
        continue;
      } else {
        return false;
      }
    }
    return true;
  };
  std::string runtime_input_shape_str =
      string::join_strings(runtime_input_shape, ',');
  std::string min_input_shape_str = string::join_strings(min_input_shape, ',');
  std::string max_input_shape_str = string::join_strings(max_input_shape, ',');
  PADDLE_ENFORCE_EQ(is_input_shape_valid(
                        runtime_input_shape, min_input_shape, max_input_shape),
                    true,
                    phi::errors::InvalidArgument(
                        "TRT runtime input shape of %s is invalid. Expect "
                        "runtime input shape to be within min/max input shape "
                        "configured in SetTRTDynamicShapeInfo(),"
                        "but got runtime input shape = [%s], min input shape = "
                        "[%s], max input shape = [%s].",
                        x,
                        runtime_input_shape_str,
                        min_input_shape_str,
                        max_input_shape_str));
}

static phi::DataType TRT2PaddleDataType(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return phi::DataType::FLOAT32;
    case nvinfer1::DataType::kINT32:
      return phi::DataType::INT32;
    case nvinfer1::DataType::kHALF:
      return phi::DataType::FLOAT16;
    case nvinfer1::DataType::kINT8:
      return phi::DataType::INT8;
#if IS_TRT_VERSION_GE(7000)
    case nvinfer1::DataType::kBOOL:
      return phi::DataType::BOOL;
#endif
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "unknown fluid datatype in Fluid op converter"));
      return phi::DataType::FLOAT32;
  }
}

void TensorRTEngineInstruction::PrepareDynamicShape() {
  // get runtime input shapes and shape tensors.
  std::map<std::string, std::vector<int32_t>> runtime_input_shape;
  std::map<std::string, std::vector<int32_t>> runtime_shape_tensor;
  const paddle::framework::Scope &scope = *(value_exec_info_->GetScope());
  pir::Value source_value = op_->operand_source(0);
  auto in_var_name = value_exec_info_->GetVarName(source_value);

  PADDLE_ENFORCE_NOT_NULL(scope.FindVar(in_var_name),
                          phi::errors::PreconditionNotMet(
                              "can not find var[%s] in scope", in_var_name));
  auto var = scope.FindVar(in_var_name);
  auto &variable_array = var->Get<VariableRefArray>();
  PADDLE_ENFORCE_EQ(variable_array.size(),
                    input_names_.size(),
                    phi::errors::InvalidArgument(
                        "Input tensor num(%d) is not equal with the input "
                        "names num(%d) in TensorRTEngineInstruction",
                        variable_array.size(),
                        input_names_.size()));
  for (size_t i = 0; i < variable_array.size(); ++i) {
    if (!variable_array[i]->IsType<phi::DenseTensor>()) {
      PADDLE_THROW(
          phi::errors::Unimplemented("Only support Vector<DenseTensor> now "
                                     "not support vector<%d>.",
                                     variable_array[i]->Type()));
    }
    auto input_tensor = variable_array[i]->Get<phi::DenseTensor>();
    auto name = input_names_[i];
    if (name == "") {
      continue;
    }

    VLOG(4) << "trt engine runtime input name(" << name << "), dims("
            << input_tensor.dims() << ")";
    auto t_shape = common::vectorize<int32_t>(input_tensor.dims());
    runtime_input_shape.insert(std::make_pair(name, t_shape));
    // We need collect value range for shape tensor for Paddle-TRT's use.
    // To be noticed, this method to identify all inputs/outputs is shape
    // tensors; After, TRT Engine gets whether it is a real shape tensor.
    auto is_shape_tensor = true;
    if (trt_engine_->engine()) {
      auto *engine = trt_engine_->engine();
      is_shape_tensor =
          engine->isShapeBinding(engine->getBindingIndex(name.c_str()));
      if (!is_shape_tensor) {
        runtime_shape_tensor.erase(name);
        VLOG(4) << "trt engine runtime delete shape name(" << name << "), dims("
                << input_tensor.dims() << ")";
      }
    }
    if ((input_tensor.dtype() == phi::DataType::INT32 ||
         input_tensor.dtype() == phi::DataType::INT64) &&
        is_shape_tensor) {
      std::vector<int> int32_host(input_tensor.numel());
      paddle::platform::DeviceContextPool &pool =
          paddle::platform::DeviceContextPool::Instance();

      if (input_tensor.place().GetType() == phi::AllocationType::CPU) {
        auto &int32_tensor = input_tensor;
        if (input_tensor.dtype() == phi::DataType::INT64) {
          auto *cpu_ctx = pool.Get(phi::CPUPlace());
          int32_tensor = phi::funcs::TransDataType(
              reinterpret_cast<const phi::CPUContext &>(*cpu_ctx),
              input_tensor,
              DataType::INT32);
        }
        phi::memory_utils::Copy(phi::CPUPlace(),
                                int32_host.data(),
                                phi::CPUPlace(),
                                int32_tensor.data<int>(),
                                int32_tensor.numel() * sizeof(int));
      } else if (input_tensor.place().GetType() == phi::AllocationType::GPU) {
#if defined(PADDLE_WITH_CUDA)
        auto *dev_ctx = pool.Get(input_tensor.place());
        auto &int32_tensor = input_tensor;
        if (input_tensor.dtype() == phi::DataType::INT64) {
          int32_tensor = phi::funcs::TransDataType(
              reinterpret_cast<const phi::GPUContext &>(*dev_ctx),
              input_tensor,
              DataType::INT32);
        }
        phi::memory_utils::Copy(phi::CPUPlace(),
                                int32_host.data(),
                                int32_tensor.place(),
                                int32_tensor.data<int>(),
                                int32_tensor.numel() * sizeof(int),
                                nullptr);
#endif
      }
      runtime_shape_tensor[name] = int32_host;
    }
  }

  if (!allow_build_at_runtime_) {
    std::map<std::string, std::vector<int>> min_input_shape =
        trt_engine_->min_input_shape();
    std::map<std::string, std::vector<int>> max_input_shape =
        trt_engine_->max_input_shape();
    for (auto x : input_names_) {
      if (x == "") {
        continue;
      }
      auto is_shape_tensor = false;
      if (trt_engine_->engine()) {
        auto *engine = trt_engine_->engine();
#if IS_TRT_VERSION_GE(8600)
        is_shape_tensor = engine->isShapeInferenceIO(x.c_str());
#else
        is_shape_tensor =
            engine->isShapeBinding(engine->getBindingIndex(x.c_str()));
#endif
      }
      if (is_shape_tensor) {
        continue;
      }
      PADDLE_ENFORCE_EQ(
          min_input_shape.count(x),
          true,
          phi::errors::InvalidArgument(
              "Input %s not found in TRT engine min_input_shape.", x));
      PADDLE_ENFORCE_EQ(
          max_input_shape.count(x),
          true,
          phi::errors::InvalidArgument(
              "Input %s not found in TRT engine max_input_shape.", x));
      RuntimeDynamicShapeCheck(
          x, runtime_input_shape[x], min_input_shape[x], max_input_shape[x]);
    }
  } else {
    // compare runtime_input_shape and trt_engine dynamic shapes.
    std::vector<std::string> shape_changed_name;
    std::vector<std::string> tensor_changed_name;
    bool is_adjusted =
        trt_engine_->AdjustDynamicShapeRange(runtime_input_shape,
                                             runtime_shape_tensor,
                                             &shape_changed_name,
                                             &tensor_changed_name);
    if (is_adjusted) {
      if (trt_engine_->engine()) {
        trt_engine_->ResetContext();
        trt_engine_->ClearTensorMap();
      }
      auto *anc = scope.parent();
      while (anc && anc->parent()) {
        anc = anc->parent();
      }
      if (anc == nullptr) {
        anc = &scope;
      }

      // TODO(YuanRisheng): Rebuild TRT Engine
      // PrepareTRTEngine(*anc, trt_engine_);

      // TODO(YuanRisheng): update global shape_range

      // TODO(YuanRisheng): If add use_static_engine_ attr, need support save
      // rebuild trt_engine
    }
  }
}

void TensorRTEngineInstruction::BindInputTensor(
    const std::string &input_name,
    const phi::DenseTensor &input_tensor,
    const Scope &scope,
    std::vector<void *> &buffers,
    std::vector<int> &shape_v,
    int *runtime_batch) {
  if (input_name == "") {
    return;
  }
  auto dev_place = dev_ctx_->GetPlace();
  const int num_bindings = trt_engine_->GetNbBindings();
  int binding_offset = 0;
  nvinfer1::IExecutionContext *trt_context = nullptr;
  // Initialize context and get offset by profile index
  trt_context = trt_engine_->context();
  binding_offset = trt_engine_->GetBindingsOffset();

  PADDLE_ENFORCE_GT(
      input_tensor.numel(),
      0,
      phi::errors::InvalidArgument(
          "The input tensor named %s of trt-subgraph must "
          "have >0 elements, but now have %d elements. "
          "It's likely that this tensor is connected to a Concat op inside "
          "a trt-subgraph, "
          "try to ues API to forbid this op into trt-subgraph.",
          input_name,
          input_tensor.numel()));

  // check the input_tensor
  if (!(input_tensor.place().GetType() == phi::AllocationType::GPU)) {
    phi::DenseTensor out;
    phi::Copy(*dev_ctx_, input_tensor, dev_place, false, &out);
    const_cast<phi::DenseTensor &>(input_tensor).ShareDataWith(out);
  }
  auto input_shape = common::vectorize<int64_t>(input_tensor.dims());

  // This must be a zero dimension tensor.
  // At present, we convert it to a 1D tensor to feed them into Trt.
  if (input_shape.empty()) {
    PADDLE_ENFORCE_EQ(input_tensor.numel(),
                      1UL,
                      phi::errors::PreconditionNotMet(
                          "This tensor must have one element, but got %ld.",
                          input_tensor.numel()));
    input_shape.push_back(1);
  }

  // Get index of profile 0 first, then plus binding offset
  const int bind_index =
      trt_engine_->engine()->getBindingIndex(input_name.c_str()) +
      binding_offset;
  PADDLE_ENFORCE_LT(bind_index,
                    num_bindings,
                    phi::errors::InvalidArgument(
                        "Wrong TRT engine input binding index. Expected The "
                        "binding index of TRT engine input to be less than "
                        "the number of inputs and outputs. Received binding "
                        "index=%d >= total inputs and outputs=%d",
                        bind_index,
                        num_bindings));

#if IS_TRT_VERSION_GE(6000)
#if IS_TRT_VERSION_GE(8500)
  if (trt_engine_->engine()->isShapeBinding(bind_index) &&
      trt_engine_->engine()->bindingIsInput(bind_index)) {
    if (input_tensor.dtype() == phi::DataType::INT32) {
      phi::memory_utils::Copy(phi::CPUPlace(),
                              shape_v.data(),
                              input_tensor.place(),
                              input_tensor.data<int32_t>(),
                              input_tensor.numel() * sizeof(int),
                              nullptr);
    } else if (input_tensor.dtype() == phi::DataType::INT64) {
      std::string x_t = input_name + "_cast_to_INT32";
      if (scope.FindVar(x_t) == nullptr) {
        const_cast<framework::Scope *>(&scope)->Var(x_t);
      }
      auto int32_tensor = scope.FindVar(x_t)->GetMutable<phi::DenseTensor>();
      *int32_tensor = phi::Cast<int64_t>(
          reinterpret_cast<const phi::GPUContext &>(*dev_ctx_),
          input_tensor,
          phi::DataType::INT32);
      phi::memory_utils::Copy(phi::CPUPlace(),
                              shape_v.data(),
                              int32_tensor->place(),
                              int32_tensor->data<int32_t>(),
                              int32_tensor->numel() * sizeof(int),
                              nullptr);
    }
    trt_context->setTensorAddress(input_name.c_str(), shape_v.data());
  } else {
    trt_context->setInputShape(
        input_name.c_str(),
        paddle::platform::Vec2TRT_Dims(input_shape, input_name, true));
  }
#else
  trt_context->setBindingDimensions(
      bind_index,
      paddle::platform::Vec2TRT_Dims(input_shape, input_name, true));
  // If this x is a shape tensor, we need call setInputShapeBinding
  if (trt_engine_->engine()->isShapeBinding(bind_index) &&
      trt_engine_->engine()->bindingIsInput(bind_index)) {
    if (input_tensor.dtype() == phi::DataType::INT32) {
      phi::memory_utils::Copy(phi::CPUPlace(),
                              shape_v.data(),
                              input_tensor.place(),
                              input_tensor.data<int32_t>(),
                              input_tensor.numel() * sizeof(int),
                              nullptr);
    } else if (input_tensor.dtype() == phi::DataType::INT64) {
      std::string x_t = input_name + "_cast_to_INT32";
      if (scope.FindVar(x_t) == nullptr) {
        const_cast<framework::Scope *>(&scope)->Var(x_t);
      }
      auto int32_tensor = scope.FindVar(x_t)->GetMutable<phi::DenseTensor>();
      *int32_tensor = phi::Cast<int64_t>(
          reinterpret_cast<const phi::GPUContext &>(*dev_ctx_),
          input_tensor,
          phi::DataType::INT32);
      phi::memory_utils::Copy(phi::CPUPlace(),
                              shape_v.data(),
                              int32_tensor->place(),
                              int32_tensor->data<int32_t>(),
                              int32_tensor->numel() * sizeof(int),
                              nullptr);
    }
    trt_context->setInputShapeBinding(bind_index, shape_v.data());
  }
#endif
#endif

  *runtime_batch = input_shape[0];
  VLOG(1) << "trt input [" << input_name << "] dtype is "
          << input_tensor.dtype();

  auto indata_type = paddle::platform::PhiType2NvType(input_tensor.dtype());
  auto intrt_index = trt_engine_->engine()->getBindingIndex(input_name.c_str());
  auto intrt_type = trt_engine_->engine()->getBindingDataType(intrt_index);
  PADDLE_ENFORCE_EQ(indata_type,
                    intrt_type,
                    phi::errors::InvalidArgument(
                        "The TRT Engine OP's input type [%d] should equal "
                        "to the input data type [%d].",
                        static_cast<int>(intrt_type),
                        static_cast<int>(indata_type)));
  if (input_tensor.dtype() == phi::DataType::FLOAT32) {
    buffers[bind_index] =
        static_cast<void *>(const_cast<float *>(input_tensor.data<float>()));
  } else if (input_tensor.dtype() == phi::DataType::FLOAT64) {
    std::string x_t = input_name + "_cast_to_FP32";
    if (scope.FindVar(x_t) == nullptr) {
      const_cast<framework::Scope *>(&scope)->Var(x_t);
    }
    auto fp32_tensor = scope.FindVar(x_t)->GetMutable<phi::DenseTensor>();
    *fp32_tensor =
        phi::Cast<double>(reinterpret_cast<const phi::GPUContext &>(*dev_ctx_),
                          input_tensor,
                          phi::DataType::FLOAT32);
    buffers[bind_index] = static_cast<void *>(fp32_tensor->data<float>());
  } else if (input_tensor.dtype() == phi::DataType::INT64) {
    std::string x_t = input_name + "_cast_to_INT32";
    if (scope.FindVar(x_t) == nullptr) {
      const_cast<framework::Scope *>(&scope)->Var(x_t);
    }
    auto int32_tensor = scope.FindVar(x_t)->GetMutable<phi::DenseTensor>();
    *int32_tensor =
        phi::Cast<int64_t>(reinterpret_cast<const phi::GPUContext &>(*dev_ctx_),
                           input_tensor,
                           phi::DataType::INT32);
    buffers[bind_index] = static_cast<void *>(int32_tensor->data<int32_t>());
  } else if (input_tensor.dtype() == phi::DataType::INT32) {
    buffers[bind_index] = static_cast<void *>(
        const_cast<int32_t *>(input_tensor.data<int32_t>()));
  } else if (input_tensor.dtype() == phi::DataType::FLOAT16) {
    buffers[bind_index] = static_cast<void *>(
        const_cast<float16 *>(input_tensor.data<float16>()));
#if IS_TRT_VERSION_GE(8400)
  } else if (input_tensor.dtype() == phi::DataType::BOOL) {
    buffers[bind_index] =
        static_cast<void *>(const_cast<bool *>(input_tensor.data<bool>()));
#endif
  } else {
    PADDLE_THROW(
        phi::errors::Fatal("The TRT Engine OP only support "
                           "float/double/int32_t/int64_t/float16/bool input."));
  }
}

void TensorRTEngineInstruction::BindOutputTensor(
    std::string output_name,
    phi::DenseTensor *output_tensor,
    int output_index,
    std::vector<void *> &buffers,
    int *runtime_batch) {
  int binding_offset = 0;
  const int num_bindings = trt_engine_->GetNbBindings();
  nvinfer1::IExecutionContext *trt_context = nullptr;
  // Initialize context and get offset by profile index
  trt_context = trt_engine_->context();
  binding_offset = trt_engine_->GetBindingsOffset();

  const int bind_index =
      trt_engine_->engine()->getBindingIndex(output_name.c_str()) +
      binding_offset;
  std::vector<int> ddim;

#if IS_TRT_VERSION_GE(8500)
  auto x_name = trt_engine_->engine()->getBindingName(bind_index);
  auto dims = trt_context->getTensorShape(x_name);
  int nb_dims = dims.nbDims;
  for (; nb_dims > 0; nb_dims--) {
    // some 'x 1' of shape is normal, no need to remove it
    if (dims.d[nb_dims - 1] != 1 || nb_dims == outputs_rank_[output_index])
      break;
  }
  for (int i = 0; i < nb_dims; i++) {
    ddim.push_back(dims.d[i]);
  }
#else
  auto dims = trt_context->getBindingDimensions(bind_index);
  int nb_dims = dims.nbDims;
  for (; nb_dims > 0; nb_dims--) {
    // some 'x 1' of shape is normal, no need to remove it
    if (dims.d[nb_dims - 1] != 1 || nb_dims == outputs_rank_[output_index])
      break;
  }
  for (int i = 0; i < nb_dims; i++) {
    ddim.push_back(dims.d[i]);
  }
#endif

  auto *fluid_t = output_tensor;
  fluid_t->Resize(common::make_ddim(ddim));

  PADDLE_ENFORCE_LT(bind_index,
                    num_bindings,
                    phi::errors::InvalidArgument(
                        "The binding index in TRT engine should be less "
                        "than the number of bindings, but got binding "
                        "index = %d, number of bindings = %d.",
                        bind_index,
                        num_bindings));
  auto trt_type = trt_engine_->engine()->getBindingDataType(bind_index);
  // get adr and set type
  VLOG(1) << "trt output [" << output_name << "] dtype is "
          << TRT2PaddleDataType(trt_type);
  buffers[bind_index] = static_cast<void *>(
      dev_ctx_->Alloc(fluid_t, TRT2PaddleDataType(trt_type)));
}

void TensorRTEngineInstruction::RunTrt() {
  auto dev_place = dev_ctx_->GetPlace();
  const paddle::framework::Scope &scope = *(value_exec_info_->GetScope());
  int runtime_batch = -1;
  auto stream = reinterpret_cast<const phi::GPUContext &>(*dev_ctx_).stream();

  // Get the total over all profiles
  const int num_bindings = trt_engine_->GetNbBindings();
  std::vector<void *> buffers(num_bindings, nullptr);

  pir::Value source_value = op_->operand_source(0);
  auto in_var_name = value_exec_info_->GetVarName(source_value);

  PADDLE_ENFORCE_NOT_NULL(scope.FindVar(in_var_name),
                          phi::errors::PreconditionNotMet(
                              "can not find var[%s] in scope", in_var_name));
  auto in_var = scope.FindVar(in_var_name);
  auto &in_variable_array = in_var->Get<VariableRefArray>();
  std::vector<std::vector<int>> shape_inputs(in_variable_array.size());
  for (size_t i = 0; i < in_variable_array.size(); ++i) {
    if (in_variable_array[i]->IsType<phi::DenseTensor>()) {
      auto input_tensor = in_variable_array[i]->Get<phi::DenseTensor>();
      // we will use shape_input when input is a shape tensor
      shape_inputs[i].resize(input_tensor.numel());
      // Bind input tensor to TRT.
      BindInputTensor(input_names_[i],
                      input_tensor,
                      scope,
                      buffers,
                      shape_inputs[i],
                      &runtime_batch);
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("Only support Vector<DenseTensor> now "
                                     "not support vector<%d>.",
                                     in_variable_array[i]->Type()));
    }
  }

  // Bind output tensor to TRT.
  VLOG(4) << "TensorRT Engine Op Outputs:";
  pir::Value result_value = op_->result(0);
  auto out_var_name = value_exec_info_->GetVarName(result_value);

  PADDLE_ENFORCE_NOT_NULL(scope.FindVar(out_var_name),
                          phi::errors::PreconditionNotMet(
                              "can not find var[%s] in scope", out_var_name));
  auto out_var = scope.FindVar(out_var_name);
  auto *out_variable_array = out_var->GetMutable<VariableRefArray>();
  for (size_t i = 0; i < out_variable_array->size(); ++i) {
    if (out_variable_array->at(i)->IsType<phi::DenseTensor>()) {
      auto output_tensor = const_cast<phi::DenseTensor *>(
          &(out_variable_array->at(i)->Get<phi::DenseTensor>()));
      // Bind input tensor to TRT.
      BindOutputTensor(
          output_names_[i], output_tensor, i, buffers, &runtime_batch);
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("Only support Vector<DenseTensor> now "
                                     "not support vector<%d>.",
                                     out_variable_array->at(i)->Type()));
    }
  }

  // Execute the engine.
  trt_engine_->Execute(runtime_batch, &buffers, stream);

  for (size_t i = 0; i < out_variable_array->size(); ++i) {
    auto type = outputs_dtype_[i];

    if (type == phi::DataType::INT64) {
      auto y = output_names_[i];
      auto *fluid_v = out_variable_array->at(i);
      auto *fluid_t =
          const_cast<phi::DenseTensor *>(&(fluid_v->Get<phi::DenseTensor>()));
      std::string y_t = y + "_cast_to_INT64";
      if (scope.FindVar(y_t) == nullptr) {
        const_cast<framework::Scope *>(&scope)->Var(y_t);
      }
      auto int32_tensor = scope.FindVar(y_t)->GetMutable<phi::DenseTensor>();
      int32_tensor->Resize(fluid_t->dims());
      dev_ctx_->Alloc<int32_t>(int32_tensor);
      phi::Copy(*dev_ctx_, *fluid_t, dev_place, false, int32_tensor);
      *fluid_t = phi::Cast<int32_t>(
          reinterpret_cast<const phi::GPUContext &>(*dev_ctx_),
          *int32_tensor,
          phi::DataType::INT64);
    } else if (type == phi::DataType::FLOAT64) {
      auto y = output_names_[i];
      auto *fluid_v = out_variable_array->at(i);
      auto *fluid_t =
          const_cast<phi::DenseTensor *>(&(fluid_v->Get<phi::DenseTensor>()));
      std::string y_t = y + "_cast_to_FP64";
      if (scope.FindVar(y_t) == nullptr) {
        const_cast<framework::Scope *>(&scope)->Var(y_t);
      }
      auto fp32_tensor = scope.FindVar(y_t)->GetMutable<phi::DenseTensor>();
      fp32_tensor->Resize(fluid_t->dims());
      dev_ctx_->Alloc<float>(fp32_tensor);
      phi::Copy(*dev_ctx_, *fluid_t, dev_place, false, fp32_tensor);
      *fluid_t =
          phi::Cast<float>(reinterpret_cast<const phi::GPUContext &>(*dev_ctx_),
                           *fp32_tensor,
                           phi::DataType::FLOAT64);
    }
  }
}

void TensorRTEngineInstruction::Run() {
  PrepareDynamicShape();
  RunTrt();
}

}  // namespace framework
}  // namespace paddle
