// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_transform.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/execution_context.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/imperative/var_helper.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/selected_rows.h"

DECLARE_bool(use_mkldnn);

namespace paddle {
namespace imperative {

const framework::Tensor* GetTensorFromVar(const framework::Variable& var);

template <typename VarType>
static void SetForwardDataTypeOfGradVar(const std::shared_ptr<VarType>& var);

template <>
void SetForwardDataTypeOfGradVar<VariableWrapper>(
    const std::shared_ptr<VariableWrapper>& var) {
  if (var->HasGradVar()) {
    auto grad_var = var->GetGradVar();
    VLOG(6) << "Set grad var (" << grad_var->Name() << ")'s forward dtype to ("
            << framework::DataTypeToString(var->DataType()) << ").";
    grad_var->SetForwardDataType(var->DataType());
  }
}

template <>
void SetForwardDataTypeOfGradVar<VarBase>(const std::shared_ptr<VarBase>& var) {
  if (var->HasGradVar()) {
    auto& shared_var = var->SharedVar();
    SetForwardDataTypeOfGradVar<VariableWrapper>(shared_var);
  }
}

template <>
void SetForwardDataTypeOfGradVar<egr::EagerVariable>(
    const std::shared_ptr<egr::EagerVariable>& var) {
  VLOG(10) << "Var in Eager dose not support SetForwardDataTypeOfGradVar: "
           << var->name();
  // TODO(jiabin): SetForwardDataType of Grad var is not supported yet in
  // EagerMode.
}

template <typename VarType>
std::shared_ptr<NameVarMap<VarType>> PrepareData(
    const framework::OperatorWithKernel& op,
    const NameVarMap<VarType>& ins,
    const framework::OpKernelType& expected_kernel_key) {
  std::shared_ptr<NameVarMap<VarType>> tmp_ins_ptr = nullptr;
  for (const auto& name_pair : ins) {
    for (size_t i = 0; i < name_pair.second.size(); ++i) {
      auto& template_var = name_pair.second[i];
      SetForwardDataTypeOfGradVar(template_var);
      const auto* tensor = GetTensorFromVar(template_var->Var());
      if (tensor && tensor->IsInitialized() && (tensor->memory_size() != 0)) {
        auto kernel_type_for_var = op.GetKernelTypeForVar(
            name_pair.first, *tensor, expected_kernel_key);
        if (!NeedTransform(kernel_type_for_var, expected_kernel_key)) {
          continue;
        } else {
          VLOG(3) << "Transform Variable " << GetNameFromVar(template_var)
                  << " from " << kernel_type_for_var << " to "
                  << expected_kernel_key;
          VLOG(3) << GetNameFromVar(template_var)
                  << " memory size is: " << tensor->memory_size();
          if (CheckCachedKey(template_var, expected_kernel_key)) {
            VLOG(3) << "Hit variable_wrapper cache: key="
                    << expected_kernel_key;
            std::shared_ptr<VariableWrapper> cache_var =
                GetCachedValue(template_var, expected_kernel_key);
            if (tmp_ins_ptr == nullptr) {
              tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
            }

            const auto* tensor = GetTensorFromVar(cache_var->Var());
            auto tmp_var =
                std::make_shared<VarType>(GetNameFromVar(template_var));
            SetType(tmp_var, GetType(template_var));
            SetTensorToVariable(
                cache_var->Var(), *tensor, tmp_var->MutableVar());
            (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;
          } else {
            framework::Tensor out;
            TransformData(
                expected_kernel_key, kernel_type_for_var, *tensor, &out);
            if (NeedTransformDataType(kernel_type_for_var,
                                      expected_kernel_key)) {
              // To avoid NameVarMap copy construction overhead in general
              // scenarios, if inplace transformed, return original input
              // directly
              if (tmp_ins_ptr == nullptr) {
                tmp_ins_ptr = std::make_shared<NameVarMap<VarType>>(ins);
              }
              auto tmp_var =
                  std::make_shared<VarType>(GetNameFromVar(template_var));
              SetType(tmp_var, GetType(template_var));
              SetTensorToVariable(
                  template_var->Var(), out, tmp_var->MutableVar());
              (*tmp_ins_ptr)[name_pair.first][i] = tmp_var;
              SetCachedValue(template_var, expected_kernel_key, tmp_var);
              VLOG(3) << "Set cache to variable_wrapper: key="
                      << expected_kernel_key;
            } else {
              // if dtype is same, transform inplace will not change the
              // original
              // value, transform inplace to avoid multiple copy
              SetTensorToVariable(
                  template_var->Var(), out, template_var->MutableVar());
            }
          }
        }
      }
    }
  }
  return tmp_ins_ptr;
}

class PreparedOp {
 public:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             const framework::OpKernelType& kernel_type,
             const framework::OperatorWithKernel::OpKernelFunc& func,
             const phi::ArgumentMappingFn* arg_map_fn,
             const phi::KernelSignature* default_kernel_signature,
             platform::DeviceContext* dev_ctx);

  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             const framework::OpKernelType& kernel_type,
             const phi::ArgumentMappingFn* arg_map_fn,
             const phi::KernelSignature* default_kernel_signature,
             phi::KernelSignature&& kernel_signature,
             const phi::Kernel& phi_kernel,
             platform::DeviceContext* dev_ctx);

  static PreparedOp Prepare(const NameVarMap<VarBase>& ins,
                            const NameVarMap<VarBase>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs,
                            const framework::AttributeMap& default_attrs);

  static PreparedOp Prepare(const NameVarMap<VariableWrapper>& ins,
                            const NameVarMap<VariableWrapper>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs,
                            const framework::AttributeMap& default_attrs);

  static PreparedOp Prepare(const NameVarMap<egr::EagerVariable>& ins,
                            const NameVarMap<egr::EagerVariable>& outs,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place,
                            const framework::AttributeMap& attrs,
                            const framework::AttributeMap& default_attrs);

  void Run(const NameVarMap<VarBase>& in,
           const NameVarMap<VarBase>& out,
           const framework::AttributeMap& attrs,
           const framework::AttributeMap& default_attrs);

  void Run(const NameVarMap<VariableWrapper>& ins,
           const NameVarMap<VariableWrapper>& outs,
           const framework::AttributeMap& attrs,
           const framework::AttributeMap& default_attrs);

  void Run(const NameVarMap<egr::EagerVariable>& ins,
           const NameVarMap<egr::EagerVariable>& outs,
           const framework::AttributeMap& attrs,
           const framework::AttributeMap& default_attrs);

  const framework::OpKernelType& kernel_type() const { return kernel_type_; }

 private:
  const framework::OperatorBase& op_;
  const framework::RuntimeContext& ctx_;
  framework::OpKernelType kernel_type_;
  framework::OperatorWithKernel::OpKernelFunc func_;
  platform::DeviceContext* dev_ctx_;
  // NOTE(chenweihang): Similar op members are used to adapt to
  // new phi kernel, if there is a better design in the future,
  // we may polish the implementation here
  bool run_phi_kernel_{false};
  bool run_kp_kernel_{false};
  const phi::ArgumentMappingFn* arg_map_fn_;
  const phi::KernelSignature* default_kernel_signature_;
  phi::KernelSignature kernel_signature_;
  const phi::Kernel& phi_kernel_;

  static const phi::KernelFactory& phi_kernel_factory;
  static const phi::OpUtilsMap& phi_op_utils_map;
  static const phi::DefaultKernelSignatureMap& default_phi_kernel_sig_map;
};

const inline framework::Attribute* GetAttr(
    const framework::AttributeMap& attrs,
    const framework::AttributeMap& default_attrs,
    const std::string& name) {
  auto it = attrs.find(name);
  bool found = it != attrs.end();
  if (!found) {
    it = default_attrs.find(name);
    found = it != default_attrs.end();
  }
  if (found) {
    return &it->second;
  }
  return nullptr;
}

template <typename VarType>
void BuildDygraphPhiKernelContext(const phi::KernelSignature& kernel_signature,
                                  const phi::Kernel& phi_kernel,
                                  const NameVarMap<VarType>& ins,
                                  const NameVarMap<VarType>& outs,
                                  const framework::AttributeMap& attrs,
                                  const framework::AttributeMap& default_attrs,
                                  platform::DeviceContext* dev_ctx,
                                  phi::KernelContext* kernel_ctx) {
  kernel_ctx->SetDeviceContext(dev_ctx);

  const auto& input_names = kernel_signature.input_names;
  const auto& attr_names = kernel_signature.attr_names;
  const auto& output_names = kernel_signature.output_names;

  auto& input_defs = phi_kernel.args_def().input_defs();
  auto& output_defs = phi_kernel.args_def().output_defs();
  auto& attr_defs = phi_kernel.args_def().attribute_defs();

  PADDLE_ENFORCE_EQ(
      input_names.size(),
      input_defs.size(),
      platform::errors::InvalidArgument(
          "Op %s: the size of inputs_args names (%d) must be equal to "
          "the size of kernel input_defs (%d).",
          kernel_signature.name,
          input_names.size(),
          input_defs.size()));

  PADDLE_ENFORCE_EQ(
      output_names.size(),
      output_defs.size(),
      platform::errors::InvalidArgument(
          "Op %s: the size of outputs_args names (%d) must be equal to "
          "the size of kernel output_defs (%d).",
          kernel_signature.name,
          output_names.size(),
          output_defs.size()));

  PADDLE_ENFORCE_EQ(
      attr_names.size(),
      attr_defs.size(),
      platform::errors::InvalidArgument(
          "Op %s: the size of attribute_args names (%d) must be equal "
          "to the size of kernel attribute_defs (%d).",
          kernel_signature.name,
          attr_names.size(),
          attr_defs.size()));

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto it = ins.find(input_names[i]);

    size_t start_idx = (i == 0 ? 0 : kernel_ctx->InputRangeAt(i - 1).second);

    if (it == ins.end()) {
      if (LIKELY(input_defs[i].type_index ==
                 std::type_index(typeid(paddle::optional<phi::DenseTensor>)))) {
        kernel_ctx->EmplaceBackInputWithoutSetRange(nullptr);
        auto end_idx = start_idx + 1;
        kernel_ctx->AssignInputRange(std::make_pair(start_idx, end_idx), i);
        continue;
      } else if (input_defs[i].type_index ==
                 std::type_index(typeid(
                     paddle::optional<std::vector<const phi::DenseTensor*>>))) {
        kernel_ctx->EmplaceBackInputWithoutSetRange(nullptr);
        auto end_idx = start_idx + 1;
        kernel_ctx->AssignInputRange(std::make_pair(start_idx, end_idx), i);
        continue;
      } else {
        PADDLE_THROW(phi::errors::NotFound(
            "Can not find input variable '%s' for %s OP, please check whether "
            "the name setting in OpArgumentMapping is consistent with that in "
            "OpMaker.",
            input_names[i],
            kernel_signature.name));
      }
    }

    auto& ins_vector = it->second;
    size_t end_idx = start_idx + ins_vector.size();

    for (size_t offset = 0; offset < ins_vector.size(); ++offset) {
      const phi::TensorBase* tensor_in = nullptr;
      auto& var = ins_vector[offset]->Var();
      if (var.template IsType<phi::DenseTensor>()) {
        tensor_in = &(var.template Get<phi::DenseTensor>());
        kernel_ctx->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var.template IsType<phi::SelectedRows>()) {
        tensor_in = &(var.template Get<phi::SelectedRows>());
        kernel_ctx->EmplaceBackInputWithoutSetRange(tensor_in);
      } else if (var.template IsType<framework::LoDTensorArray>()) {
        paddle::small_vector<const phi::TensorBase*> tensor_vector;
        auto& tensor_array = var.template Get<framework::LoDTensorArray>();
        for (auto& t : tensor_array) {
          tensor_vector.emplace_back(&t);
        }
        kernel_ctx->EmplaceBackInputsWithoutSetRange(tensor_vector);
        end_idx += tensor_array.size() - 1;
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported input `%s` type when call pt kernel.",
            framework::ToTypeName(var.Type())));
      }
    }
    kernel_ctx->AssignInputRange(std::make_pair(start_idx, end_idx), i);
  }
  VLOG(6) << "BuildDygraphPhiKernelContext: Inputs parsing completed.";

  for (size_t i = 0; i < output_names.size(); ++i) {
    size_t start_idx = (i == 0 ? 0 : kernel_ctx->OutputRangeAt(i - 1).second);

    auto iter = outs.find(output_names[i]);
    if (iter == outs.end()) {
      kernel_ctx->EmplaceBackOutputWithoutSetRange({nullptr});
      kernel_ctx->AssignOutputRange(std::make_pair(start_idx, start_idx + 1),
                                    i);
      continue;
    }

    auto& outs_vector = iter->second;
    size_t end_idx = start_idx + outs_vector.size();

    for (size_t offset = 0; offset < outs_vector.size(); ++offset) {
      if (outs_vector[offset] == nullptr) {
        kernel_ctx->EmplaceBackOutputWithoutSetRange({nullptr});
        continue;
      }

      phi::TensorBase* tensor_out = nullptr;
      auto* var = outs_vector[offset]->MutableVar();
      if (var) {
        if (var->template IsType<phi::DenseTensor>()) {
          tensor_out = var->template GetMutable<phi::DenseTensor>();
          kernel_ctx->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<phi::SelectedRows>()) {
          tensor_out = var->template GetMutable<phi::SelectedRows>();
          kernel_ctx->EmplaceBackOutputWithoutSetRange(tensor_out);
        } else if (var->template IsType<framework::LoDTensorArray>()) {
          paddle::small_vector<phi::TensorBase*> tensor_vector;
          auto* tensor_array =
              var->template GetMutable<framework::LoDTensorArray>();
          for (auto& t : *tensor_array) {
            tensor_vector.emplace_back(&t);
          }
          kernel_ctx->EmplaceBackOutputsWithoutSetRange(tensor_vector);
          end_idx += tensor_array->size() - 1;
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported output `%s` type when call pt kernel.",
              framework::ToTypeName(var->Type())));
        }
      } else {
        kernel_ctx->EmplaceBackOutputWithoutSetRange(tensor_out);
      }
    }
    kernel_ctx->AssignOutputRange(std::make_pair(start_idx, end_idx), i);
  }
  VLOG(6) << "BuildDygraphPhiKernelContext: Outputs parsing completed.";

  for (size_t i = 0; i < attr_names.size(); ++i) {
    VLOG(6) << "BuildDygraphPhiKernelContext: " << attr_names[i] << ": "
            << attr_defs[i].type_index;
    auto* attr_ptr = GetAttr(attrs, default_attrs, attr_names[i]);
    switch (attr_defs[i].type_index) {
      case phi::AttributeType::SCALAR:
        if (attr_ptr) {
          // scalar is in the attribute
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::FLOAT:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(float, attr))));
              break;
            case framework::proto::AttrType::FLOAT64:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(double, attr))));
              break;
            case framework::proto::AttrType::INT:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(int, attr))));
              break;
            case framework::proto::AttrType::LONG:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(int64_t, attr))));
              break;
            case framework::proto::AttrType::STRING:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(std::string, attr))));
              break;
            case framework::proto::AttrType::BOOLEAN:
              kernel_ctx->EmplaceBackAttr(
                  std::move(phi::Scalar(PADDLE_GET_CONST(bool, attr))));
              break;
            default:
              PADDLE_THROW(platform::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to Scalar when construct "
                  "KernelContext in dygraph.",
                  attr_names[i]));
          }
        } else {  // scalar is in the input
          auto& ins_vector = ins.at(attr_names[i]);
          kernel_ctx->EmplaceBackAttr(std::move(
              experimental::MakePhiScalarFromVar(ins_vector[0]->Var())));
        }
        break;
      case phi::AttributeType::INT_ARRAY:
        if (attr_ptr) {
          auto& attr = *attr_ptr;
          switch (AttrTypeID(attr)) {
            case framework::proto::AttrType::INTS:
              kernel_ctx->EmplaceBackAttr(std::move(
                  phi::IntArray(PADDLE_GET_CONST(std::vector<int32_t>, attr))));
              break;
            case framework::proto::AttrType::LONGS:
              kernel_ctx->EmplaceBackAttr(std::move(
                  phi::IntArray(PADDLE_GET_CONST(std::vector<int64_t>, attr))));
              break;
            case framework::proto::AttrType::INT:
              kernel_ctx->EmplaceBackAttr(std::move(
                  phi::IntArray(&PADDLE_GET_CONST(int32_t, attr), 1)));
              break;
            case framework::proto::AttrType::LONG:
              kernel_ctx->EmplaceBackAttr(std::move(
                  phi::IntArray(&PADDLE_GET_CONST(int64_t, attr), 1)));
              break;
            default:
              PADDLE_THROW(platform::errors::Unimplemented(
                  "Unsupported cast op attribute `%s` to IntArray when "
                  "construct KernelContext.",
                  attr_names[i]));
          }
        } else {  // shape is in the input
          auto& ins_vector = ins.at(attr_names[i]);
          if (ins_vector.size() == 1) {  // ShapeTensor
            kernel_ctx->EmplaceBackAttr(std::move(
                experimental::MakePhiIntArrayFromVar(ins_vector[0]->Var())));
          } else {  // ShapeTensorList
            std::vector<framework::Variable*> variables;
            variables.reserve(ins_vector.size());
            for (const auto& var_base : ins_vector) {
              variables.push_back(var_base->MutableVar());
            }
            kernel_ctx->EmplaceBackAttr(
                std::move(experimental::MakePhiIntArrayFromVarList(variables)));
          }
        }
        break;
      case phi::AttributeType::SCALARS: {
        PADDLE_ENFORCE_NOT_NULL(
            attr_ptr,
            platform::errors::NotFound("(%s) is not found in AttributeMap when "
                                       "buildind dygraph KernelContext.",
                                       attr_names[i]));
        auto& attr = *attr_ptr;
        switch (AttrTypeID(attr)) {
          case framework::proto::AttrType::INTS: {
            const auto& vec = PADDLE_GET_CONST(std::vector<int32_t>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case framework::proto::AttrType::LONGS: {
            const auto& vec = PADDLE_GET_CONST(std::vector<int64_t>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case framework::proto::AttrType::FLOATS: {
            const auto& vec = PADDLE_GET_CONST(std::vector<float>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case framework::proto::AttrType::FLOAT64S: {
            const auto& vec = PADDLE_GET_CONST(std::vector<double>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          case framework::proto::AttrType::BOOLEANS: {
            const auto& vec = PADDLE_GET_CONST(std::vector<bool>, attr);
            std::vector<phi::Scalar> scalar_list;
            scalar_list.reserve(vec.size());
            for (const auto& val : vec) {
              scalar_list.emplace_back(val);
            }
            kernel_ctx->EmplaceBackAttr(std::move(scalar_list));
          } break;
          default:
            PADDLE_THROW(platform::errors::Unimplemented(
                "Unsupported cast op attribute `%s` to vector<Scalar> when "
                "construct KernelContext.",
                attr_names[i]));
        }
      } break;
      default: {
        PADDLE_ENFORCE_NOT_NULL(
            attr_ptr,
            platform::errors::NotFound("(%s) is not found in AttributeMap when "
                                       "buildind dygraph KernelContext.",
                                       attr_names[i]));
        auto& attr = *attr_ptr;
        switch (attr_defs[i].type_index) {
          case phi::AttributeType::FLOAT32:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(float, attr));
            break;
          case phi::AttributeType::FLOAT64:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(double, attr));
            break;
          case phi::AttributeType::INT32:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(int, attr));
            break;
          case phi::AttributeType::BOOL:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(bool, attr));
            break;
          case phi::AttributeType::INT64:
            kernel_ctx->EmplaceBackAttr(PADDLE_GET_CONST(int64_t, attr));
            break;
          case phi::AttributeType::INT32S:
            kernel_ctx->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<int>, attr));
            break;
          case phi::AttributeType::DATA_TYPE: {
            auto data_type = framework::TransToPhiDataType(
                static_cast<framework::proto::VarType::Type>(
                    PADDLE_GET_CONST(int, attr)));
            kernel_ctx->EmplaceBackAttr(data_type);
          } break;
          case phi::AttributeType::STRING:
            kernel_ctx->EmplaceBackAttr(
                std::move(PADDLE_GET_CONST(std::string, attr)));
            break;
          case phi::AttributeType::INT64S: {
            switch (AttrTypeID(attr)) {
              case framework::proto::AttrType::LONGS:
                kernel_ctx->EmplaceBackAttr(
                    PADDLE_GET_CONST(std::vector<int64_t>, attr));
                break;
              case framework::proto::AttrType::INTS: {
                const auto& vector_int_attr =
                    PADDLE_GET_CONST(std::vector<int>, attr);
                const std::vector<int64_t> vector_int64_attr(
                    vector_int_attr.begin(), vector_int_attr.end());
                kernel_ctx->EmplaceBackAttr(vector_int64_attr);
              } break;
              default:
                PADDLE_THROW(platform::errors::Unimplemented(
                    "Unsupported cast op attribute `%s` to vector<int64_t> "
                    "when "
                    "construct KernelContext.",
                    attr_names[i]));
            }
          } break;
          case phi::AttributeType::FLOAT32S:
            kernel_ctx->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<float>, attr));
            break;
          case phi::AttributeType::STRINGS:
            kernel_ctx->EmplaceBackAttr(
                PADDLE_GET_CONST(std::vector<std::string>, attr));
            break;
          default:
            PADDLE_THROW(platform::errors::Unimplemented(
                "Unsupported cast op attribute `%s` when construct "
                "KernelContext in dygraph.",
                attr_names[i]));
        }
      }
    }
  }
  VLOG(6) << "BuildDygraphPhiKernelContext: Attributes parsing completed.";
}

template <typename VarType>
void PreparePhiData(const phi::Kernel& phi_kernel,
                    const phi::KernelSignature& kernel_signature,
                    const NameVarMap<VarType>& ins) {
  const auto& input_names = kernel_signature.input_names;
  auto& input_defs = phi_kernel.args_def().input_defs();

  PADDLE_ENFORCE_EQ(input_names.size(),
                    input_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of inputs_args names (%d) must be equal to "
                        "the size of kernel input_defs (%d).",
                        input_names.size(),
                        input_defs.size()));

  for (size_t i = 0; i < input_names.size(); ++i) {
    auto& in_def = input_defs.at(i);
    auto iter = ins.find(input_names[i]);
    if (iter == ins.end()) {
      continue;
    }
    auto& ins_vector = iter->second;

    for (size_t offset = 0; offset < ins_vector.size(); ++offset) {
      auto& var = ins_vector[offset];
      const auto* tensor_in = GetTensorFromVar(var->Var());
      if (tensor_in && tensor_in->IsInitialized() &&
          (tensor_in->memory_size() != 0)) {
        if (in_def.backend == phi::Backend::ALL_BACKEND) {
          continue;
        }
        auto tensor_backend = phi::TransToPhiBackend(tensor_in->place());
        if (in_def.backend == tensor_backend ||
            (in_def.backend == phi::Backend::GPUDNN &&
             tensor_backend == phi::Backend::GPU)) {
          continue;
        }

        auto expected_place = phi::TransToPhiPlace(in_def.backend);

        VLOG(3) << "Phi Transform Variable " << input_names[i] << " from "
                << tensor_in->place() << " to " << expected_place;

        framework::Tensor tmp_tensor;
        framework::TensorCopySync(*tensor_in, expected_place, &tmp_tensor);

        SetTensorToVariable(var->Var(), tmp_tensor, var->MutableVar());
      }
    }
  }
}

}  // namespace imperative
}  // namespace paddle
