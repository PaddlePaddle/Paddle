// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/custom_kernel_instruction.h"
#include "paddle/fluid/framework/custom_operator_utils.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

namespace paddle {
namespace framework {

void CustomKernelInstruction::BuildCustomContext(
    const paddle::dialect::OpYamlInfoParser& op_yaml_info) {
  Scope* inner_scope = value_exec_info_.GetScope();
  VLOG(6) << "Build custom op infermeta param inner_scope[" << inner_scope
          << "]";

  auto attr_map = op_->attributes();

  // EmplaceBackInputs
  auto& vec_input_tensor_params = op_yaml_info.TensorParams(true);
  auto& name2id = op_yaml_info.InputName2Id();
  for (auto& t : vec_input_tensor_params) {
    PADDLE_ENFORCE_EQ(
        name2id.count(t),
        true,
        phi::errors::NotFound("param [%s] MUST in name2id map", t));

    pir::Value ptr = op_->operand_source(op_yaml_info.InputName2Id().at(t));

    if (!IsInvalid(ptr)) {
      if (op_yaml_info.GetInputType(op_yaml_info.InputName2Id().at(t)) ==
          "pir::VectorType<paddle::dialect::DenseTensorType>") {
        vec_input_shapes_.emplace_back();
        vec_input_dtypes_.emplace_back();
        // NOTE(YuanRisheng): In dygraph mode, we can not distinguish Tensor and
        // vector<Tensor> when user inputs None, so dygraph mode appends one
        // un-initialized Tensor to CustomOpKernelContext. To be compatible with
        // dygraph mode, `custom_vec_in` also emplace_back one un-initialized
        // tensor here.
        std::vector<paddle::Tensor> custom_vec_in;
        custom_vec_in.emplace_back(paddle::Tensor());
        custom_kernel_ctx_.EmplaceBackInputs(std::move(custom_vec_in));
      } else {
        input_shapes_.emplace_back();
        input_dtypes_.emplace_back();
        custom_kernel_ctx_.EmplaceBackInput(std::move(paddle::Tensor()));
      }
      VLOG(8) << "ctx->EmplaceBackInput : an optioanl input " << t;
      continue;
    }

    auto in_var_name = value_exec_info_.GetVarName(ptr);
    VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;

    PADDLE_ENFORCE_NOT_NULL(inner_scope->FindVar(in_var_name),
                            phi::errors::PreconditionNotMet(
                                "can not find var[%s] in scope", in_var_name));
    auto var = inner_scope->FindVar(in_var_name);
    if (var->IsType<phi::DenseTensor>()) {
      auto dense_tensor_in = var->GetMutable<phi::DenseTensor>();
      std::shared_ptr<phi::DenseTensor> tensor_in(
          dense_tensor_in, [](phi::DenseTensor* ptr) {
            VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
          });
      input_shapes_.push_back(phi::vectorize(tensor_in->dims()));
      input_dtypes_.push_back(tensor_in->dtype());
      paddle::Tensor custom_in;
      custom_in.set_impl(tensor_in);
      custom_kernel_ctx_.EmplaceBackInput(std::move(custom_in));
    } else if (var->IsType<VariableRefArray>()) {
      std::vector<std::vector<int64_t>> vec_input_shape;
      std::vector<phi::DataType> vec_input_dtype;
      std::vector<paddle::Tensor> vec_custom_in;
      auto& variable_array = var->Get<VariableRefArray>();
      for (size_t i = 0; i < variable_array.size(); ++i) {
        if (variable_array[i]->IsType<phi::DenseTensor>()) {
          phi::DenseTensor* dense_tensor_in = const_cast<phi::DenseTensor*>(
              &(variable_array[i]->Get<phi::DenseTensor>()));
          std::shared_ptr<phi::DenseTensor> tensor_in(
              dense_tensor_in, [](phi::DenseTensor* ptr) {
                VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
              });
          vec_input_shape.push_back(phi::vectorize(tensor_in->dims()));
          vec_input_dtype.push_back(tensor_in->dtype());
          paddle::Tensor custom_in;
          custom_in.set_impl(tensor_in);
          vec_custom_in.push_back(std::move(custom_in));
        } else {
          PADDLE_THROW(phi::errors::Unimplemented(
              "Only support Vector<DenseTensor> and vector<SelectedRows> now, "
              "not support vector<%d>.",
              variable_array[i]->Type()));
        }
      }
      vec_input_shapes_.push_back(vec_input_shape);
      vec_input_dtypes_.push_back(vec_input_dtype);
      custom_kernel_ctx_.EmplaceBackInputs(vec_custom_in);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Not support var type [%d] ",
                                              var->Type()));
    }
  }

  // EmplaceBackAttributes
  auto& vec_attr_params = op_yaml_info.AttrParams(true);
  for (auto& t : vec_attr_params) {
    PADDLE_ENFORCE_NE(
        attr_map.find(t),
        attr_map.end(),
        phi::errors::NotFound("Not found %s in attr_map, it maybe need mapping "
                              "it in OpTranslator.",
                              t));
    auto& attr_type_name = op_yaml_info.AttrTypeName(t);
    if (attr_type_name == "pir::Int32Attribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::Int32Attribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::Int32Attribute>().data());
    } else if (attr_type_name == "pir::Int64Attribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::Int64Attribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::Int64Attribute>().data());
    } else if (attr_type_name == "pir::FloatAttribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::FloatAttribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::FloatAttribute>().data());
    } else if (attr_type_name == "pir::DoubleAttribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::DoubleAttribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::DoubleAttribute>().data());
    } else if (attr_type_name == "pir::BoolAttribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::BoolAttribute>().data());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::BoolAttribute>().data());
    } else if (attr_type_name == "pir::StrAttribute") {
      custom_attrs_.push_back(
          attr_map[t].dyn_cast<pir::StrAttribute>().AsString());
      custom_kernel_ctx_.EmplaceBackAttr(
          attr_map[t].dyn_cast<pir::StrAttribute>().AsString());
    } else if (attr_type_name == "pir::ArrayAttribute<pir::Int32Attribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();
      std::vector<int32_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<pir::Int32Attribute>(),
            true,
            phi::errors::Unimplemented(
                "the 0th elementwise MUST be pir::Int32Attribute"));
        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::Int32Attribute>().data());
        }
      }
      custom_attrs_.push_back(vec_res);
      custom_kernel_ctx_.EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::FloatAttribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();
      std::vector<float> vec_res;
      if (array_list.size() > 0) {
        if (array_list[0].isa<pir::FloatAttribute>()) {
          for (size_t i = 0; i < array_list.size(); ++i) {
            vec_res.push_back(
                array_list[i].dyn_cast<pir::FloatAttribute>().data());
          }

        } else {
          PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                  attr_type_name));
        }
      }
      custom_attrs_.push_back(vec_res);
      custom_kernel_ctx_.EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::Int64Attribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();

      std::vector<int64_t> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<pir::Int64Attribute>(),
            true,
            phi::errors::PreconditionNotMet(
                "Element in array list MUST be pir::Int64Attribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::Int64Attribute>().data());
        }
      }
      custom_attrs_.push_back(vec_res);
      custom_kernel_ctx_.EmplaceBackAttr(vec_res);
    } else if (attr_type_name == "pir::ArrayAttribute<pir::StrAttribute>") {
      auto array_list = attr_map[t].dyn_cast<pir::ArrayAttribute>().AsVector();

      std::vector<std::string> vec_res;
      if (array_list.size() > 0) {
        PADDLE_ENFORCE_EQ(
            array_list[0].isa<pir::StrAttribute>(),
            true,
            phi::errors::PreconditionNotMet(
                "Element in array list MUST be pir::StrAttribute "));

        for (size_t i = 0; i < array_list.size(); ++i) {
          vec_res.push_back(
              array_list[i].dyn_cast<pir::StrAttribute>().AsString());
        }
      }
      custom_attrs_.push_back(vec_res);
      custom_kernel_ctx_.EmplaceBackAttr(vec_res);

    } else {
      PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                              attr_type_name));
    }
    VLOG(6) << "ctx->EmplaceBackAttr: " << t;
  }

  // EmplaceBackOutputs
  VLOG(8) << "ctx->EmplaceBackOutput: ";
  for (size_t i = 0; i < op_->num_results(); ++i) {
    pir::Value out_ptr = op_->result(i);
    if (!IsInvalid(out_ptr)) {
      if (op_yaml_info.GetOutputType(i) ==
          "pir::VectorType<paddle::dialect::DenseTensorType>") {
        std::vector<paddle::Tensor> custom_vec_out;
        custom_vec_out.emplace_back();
        cache_out_ptrs_.emplace_back(nullptr);
        custom_kernel_ctx_.EmplaceBackOutputs(std::move(custom_vec_out));
      } else {
        cache_out_ptrs_.emplace_back(nullptr);
        custom_kernel_ctx_.EmplaceBackOutput(std::move(paddle::Tensor()));
      }
      VLOG(8) << "ctx->EmplaceBackOutput : an optioanl output";
      continue;
    }

    if (out_ptr.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
      auto dense_tensor_out =
          inner_scope->FindVar(value_exec_info_.GetVarName(out_ptr))
              ->GetMutable<phi::DenseTensor>();
      cache_out_ptrs_.push_back(dense_tensor_out);
      std::shared_ptr<phi::DenseTensor> tensor_out(
          dense_tensor_out, [](phi::DenseTensor* ptr) {
            VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
          });
      paddle::Tensor custom_out;
      // here only can copy the output tensor into context
      custom_out.set_impl(tensor_out);

      custom_kernel_ctx_.EmplaceBackOutput(std::move(custom_out));
      VLOG(8) << "ctx->EmplaceBackOutput DenseTensor: "
              << value_exec_info_.GetVarName(out_ptr);
    } else if (out_ptr.type().isa<pir::VectorType>()) {
      std::vector<paddle::Tensor> vec_custom_out;
      auto& variable_array =
          inner_scope->FindVar(value_exec_info_.GetVarName(out_ptr))
              ->Get<VariableRefArray>();
      std::vector<paddle::Tensor> custom_vec_out;
      for (size_t i = 0; i < variable_array.size(); ++i) {
        if (variable_array[i]->IsType<phi::DenseTensor>()) {
          auto dense_tensor_out = const_cast<phi::DenseTensor*>(
              &(variable_array[i]->Get<phi::DenseTensor>()));
          cache_out_ptrs_.emplace_back(dense_tensor_out);
          std::shared_ptr<phi::DenseTensor> tensor_out(
              dense_tensor_out, [](phi::DenseTensor* ptr) {
                VLOG(6) << ptr << " ptr will not be deleted by shared_ptr";
              });
          paddle::Tensor custom_out;
          custom_out.set_impl(tensor_out);
          custom_vec_out.push_back(std::move(custom_out));
        } else {
          PADDLE_THROW(phi::errors::Unimplemented(
              "Only support Vector<DenseTensor> and vector<SelectedRows> now, "
              "not support vector<%d>.",
              variable_array[i]->Type()));
        }
      }
      VLOG(8) << "ctx->EmplaceBackOutput VariableRefArray: "
              << value_exec_info_.GetVarName(out_ptr);
      custom_kernel_ctx_.EmplaceBackOutputs(custom_vec_out);
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("only support DenseTensor and vector "));
    }
  }
  auto& op_inputs = OpMetaInfoHelper::GetInputs(*custom_op_meta_);
  auto& op_outputs = OpMetaInfoHelper::GetOutputs(*custom_op_meta_);
  auto& op_inplace_map = OpMetaInfoHelper::GetInplaceMap(*custom_op_meta_);
  // handle inplace map
  custom_kernel_ctx_.UpdatePlainOutputs(op_inputs, op_outputs, op_inplace_map);
  VLOG(6) << "Done build custom context";
}

CustomKernelInstruction::CustomKernelInstruction(
    size_t id,
    const platform::Place& place,
    pir::Operation* op,
    const ValueExecutionInfo& value_exec_info)
    : InstructionBase(id, place), value_exec_info_(value_exec_info) {
  auto op_attributes = op->attributes();
  auto op_name =
      op_attributes.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  op_ = op;
  custom_op_name_ = op_name;
  VLOG(6) << "construct custom kernel instruction for: " << custom_op_name_;

  VLOG(6) << "finish process dist attributes";

  SetKernelType(AnalyseOpFuncType(op, place));
  VLOG(6) << "finish process analyse kernel type";

  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      yaml_interface,
      phi::errors::PreconditionNotMet(
          "can not find OpYamlInfoInterface from [%s]", custom_op_name_));
  paddle::dialect::OpYamlInfoParser yaml_info_parser(
      yaml_interface->get_op_info_(custom_op_name_),
      paddle::dialect::IsLegacyOp(custom_op_name_));
  VLOG(6) << "finish process yaml_info_parser";

  const auto& op_meta =
      paddle::framework::detail::GetOpInfoByPirName(custom_op_name_);
  custom_op_meta_ = &op_meta;
  infershape_func_ = OpMetaInfoHelper::GetInferShapeFn(op_meta);
  inferdtype_func_ = OpMetaInfoHelper::GetInferDtypeFn(op_meta);
  kernel_func_ = OpMetaInfoHelper::GetKernelFn(op_meta);
  BuildCustomContext(yaml_info_parser);
  VLOG(6) << "finish process custom context";
  auto kernel_key = op_attributes.at("kernel_key")
                        .dyn_cast<paddle::dialect::KernelAttribute>()
                        .data();
  SetDeviceContext(
      ParseDeviceContext(op,
                         phi::DeviceContextPool::Instance().Get(
                             phi::TransToPhiPlace(kernel_key.backend())),
                         place,
                         GetExecutionStream(),
                         GetStreamPriority()));
  VLOG(6) << "finish process device context";

  InitInputsOutputsIds(op, value_exec_info_);
  VLOG(6) << "finish process inputs outputs index";

  auto& no_need_buffer_ids = yaml_info_parser.NoNeedBufferIds();
  std::unordered_set<pir::Value> no_need_buffer_values;
  for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
    no_need_buffer_values.insert(op->operand_source(no_need_buffer_ids[id]));
  }
  SetNoNeedBuffer(no_need_buffer_values);
  VLOG(6) << "finish process no need buffer";
}

void CustomKernelInstruction::UpdateOutputMeta(
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<DataType>& output_dtypes) {
  PADDLE_ENFORCE_EQ(
      output_shapes.size(),
      cache_out_ptrs_.size(),
      phi::errors::InvalidArgument(
          "The number of output shapes after running custom operator's "
          "InferShapeFunc is wrong, "
          "expected contains %d Tensors' shape, but actually contains %d "
          "Tensors' shape",
          cache_out_ptrs_.size(),
          output_shapes.size()));

  PADDLE_ENFORCE_EQ(
      output_dtypes.size(),
      cache_out_ptrs_.size(),
      phi::errors::InvalidArgument(
          "The number of output dtypes after running custom operator's "
          "InferDtypeFunc is wrong, "
          "expected contains %d Tensors' dtype, but actually contains %d "
          "Tensors' dtype",
          cache_out_ptrs_.size(),
          output_dtypes.size()));

  for (size_t i = 0; i < cache_out_ptrs_.size(); ++i) {
    auto out_in_scope = cache_out_ptrs_.at(i);
    // update dims and dtype
    auto out_meta = phi::DenseTensorUtils::GetMutableMeta(out_in_scope);
    out_meta->dims = phi::make_ddim(output_shapes[i]);
    out_meta->dtype = output_dtypes[i];
  }
}

void CustomKernelInstruction::Run() {
  VLOG(3) << "Custom Operator: InferShape - calc output ddim.";
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<phi::DataType> output_dtypes;
  if (infershape_func_) {
    output_shapes =
        infershape_func_(input_shapes_, vec_input_shapes_, custom_attrs_);
  } else {
    PADDLE_ENFORCE_EQ(
        OpMetaInfoHelper::GetInputs(*custom_op_meta_).size(),
        1UL,
        phi::errors::Unavailable(
            "Your custom operator contains multiple inputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferShapeFn. "
            "At this time, the input shape will be directly set to "
            "the output shape.\n"
            "Please set the InferShapeFn of custom "
            "operator by .SetInferShapeFn(PD_INFER_SHAPE(...))"));
    PADDLE_ENFORCE_EQ(
        OpMetaInfoHelper::GetOutputs(*custom_op_meta_).size(),
        1UL,
        phi::errors::Unavailable(
            "Your custom operator contains multiple outputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferShapeFn. "
            "At this time, the input shape will be directly set to "
            "the output shape.\n"
            "Please set the InferShapeFn of custom "
            "operator by .SetInferShapeFn(PD_INFER_SHAPE(...))"));

    VLOG(3) << "Custom Operator: Default InferShape - share ddim.";
    if (input_shapes_.size() == 1) {
      output_shapes = input_shapes_;
    } else if (vec_input_shapes_.size() == 1) {
      output_shapes = vec_input_shapes_[0];
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "We only allow a custom operator that contains only one input "
          "and only one output without setting the InferShapeFn. "));
    }
  }

  if (inferdtype_func_) {
    output_dtypes =
        inferdtype_func_(input_dtypes_, vec_input_dtypes_, custom_attrs_);
  } else {
    PADDLE_ENFORCE_EQ(
        OpMetaInfoHelper::GetInputs(*custom_op_meta_).size(),
        1UL,
        phi::errors::Unavailable(
            "Your custom operator contains multiple inputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferDtypeFn. "
            "At this time, the input dtype will be directly set to "
            "the output dtype.\n"
            "Please set the InferDtypeFn of custom "
            "operator by `.SetInferDtypeFn(PD_INFER_DTYPE(...))`"));
    PADDLE_ENFORCE_EQ(
        OpMetaInfoHelper::GetOutputs(*custom_op_meta_).size(),
        1UL,
        phi::errors::Unavailable(
            "Your custom operator contains multiple outputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferDtypeFn. "
            "At this time, the input dtype will be directly set to "
            "the output dtype.\n"
            "Please set the InferDtypeFn of custom "
            "operator by `.SetInferDtypeFn(PD_INFER_DTYPE(...))`"));

    VLOG(3) << "Custom Operator: InferDtype - share dtype.";
    if (input_dtypes_.size() == 1) {
      output_dtypes = input_dtypes_;
    } else if (vec_input_dtypes_.size() == 1) {
      output_dtypes = vec_input_dtypes_[0];
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "We only allow a custom operator that contains only one input "
          "and only one output without setting the InferDtypeFn. "));
    }
  }
  UpdateOutputMeta(output_shapes, output_dtypes);

  VLOG(6) << "Run custom op " << custom_op_name_ << " kernel.";
  kernel_func_(&custom_kernel_ctx_);
  custom_kernel_ctx_.AssignInplaceOutputs();
}
}  // namespace framework
}  // namespace paddle
