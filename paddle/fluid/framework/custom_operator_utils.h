/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_tools.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace framework {
constexpr char kCustomDialectPrefix[] = "custom_op.";  // NOLINT
constexpr char kGradSuffix[] = "_grad";                // NOLINT
constexpr char kDoubleGradSuffix[] = "_grad_grad";     // NOLINT

namespace detail {

// dynamic lib load func
template <typename T>
static T* DynLoad(void* handle, std::string name) {
  T* func = reinterpret_cast<T*>(dlsym(handle, name.c_str()));
#if !defined(_WIN32)
  auto errorno = dlerror();
#else
  auto errorno = GetLastError();
#endif  // !_WIN32
  PADDLE_ENFORCE_NOT_NULL(
      func,
      common::errors::NotFound(
          "Failed to load dynamic operator library, error message(%s).",
          errorno));
  return func;
}

inline static bool IsDuplicableVar(const std::string& var_name) {
  std::string suffix = kTensorVectorSuffix;
  return var_name.rfind(suffix) != std::string::npos;
}

inline static bool IsOptionalVar(const std::string& var_name) {
  std::string suffix = kOptionalSuffix;
  return var_name.rfind(suffix) != std::string::npos;
}

inline static std::string NoGrad(const std::string& var_name,
                                 bool is_double_grad = false) {
  std::string suffix = kGradVarSuffix;
  std::string new_out_suffix = kDoubleGradNewOutSuffix;
  std::string tmp_var_name(var_name);
  if (is_double_grad &&
      (tmp_var_name.rfind(new_out_suffix) != std::string::npos)) {
    tmp_var_name = tmp_var_name.substr(
        0, tmp_var_name.size() - /*kDoubleGradNewOutSuffix length*/ 4);
  }
  return tmp_var_name.substr(0, tmp_var_name.size() - kGradVarSuffixSize);
}

inline static bool IsGradVar(const std::string& var_name, bool is_double_grad) {
  std::string suffix = kGradVarSuffix;
  if (!is_double_grad) {
    return var_name.rfind(suffix) != std::string::npos;
  } else {
    // for double grad cases, the X@GRAD is not a grad var, X@GRAD@GRAD is a
    // grad var, here we remove a @GRAD suffix
    return NoGrad(var_name).rfind(suffix) != std::string::npos;
  }
}

inline static bool IsMemberOf(const std::vector<std::string>& vec,
                              const std::string& name) {
  return std::find(vec.cbegin(), vec.cend(), name) != vec.cend();
}

inline static const OpMetaInfo* GetGradOpInfoByFwdPirName(
    const std::string& pir_op_name) {
  auto custom_name = pir_op_name.substr(strlen(kCustomDialectPrefix));
  int pos = custom_name.length();

  if (custom_name[pos - 1] == '_') {
    // deal with inplace name
    custom_name = custom_name.substr(0, pos - 1);
  }

  pos = custom_name.length();
  if (custom_name.find(kDoubleGradSuffix) != custom_name.npos) {
    pos = custom_name.find(kDoubleGradSuffix);
  } else if (custom_name.find(kGradSuffix) != custom_name.npos) {
    pos = custom_name.find(kGradSuffix);
  }
  auto custom_name_prefix = custom_name.substr(0, pos);
  auto map_iter =
      paddle::OpMetaInfoMap::Instance().GetMap().find(custom_name_prefix);
  if (map_iter == paddle::OpMetaInfoMap::Instance().GetMap().end()) {
    PADDLE_THROW("The info of custom op : " + custom_name + " is not exists!");
  }
  const auto& vec_op_meta = map_iter->second;
  const OpMetaInfo* ret = nullptr;
  if (custom_name.find(kDoubleGradSuffix) != custom_name.npos) {
    PADDLE_THROW("Custom op : " + custom_name_prefix +
                 " doesn't support triple grad.");
  } else if (custom_name.find(kGradSuffix) != custom_name.npos) {
    bool has_double_grad = vec_op_meta.size() >= 3;
    ret = has_double_grad ? &(vec_op_meta[2]) : nullptr;
  } else {
    bool has_grad = vec_op_meta.size() >= 2;
    ret = has_grad ? &(vec_op_meta[1]) : nullptr;
  }
  return ret;
}

inline static const OpMetaInfo& GetOpInfoByPirName(
    const std::string& pir_op_name) {
  auto custom_name = pir_op_name.substr(strlen(kCustomDialectPrefix));
  int pos = custom_name.length();

  if (custom_name[pos - 1] == '_') {
    // deal with inplace name
    custom_name = custom_name.substr(0, pos - 1);
  }

  pos = custom_name.length();
  if (custom_name.find(kDoubleGradSuffix) != custom_name.npos) {
    pos = custom_name.find(kDoubleGradSuffix);
  } else if (custom_name.find(kGradSuffix) != custom_name.npos) {
    pos = custom_name.find(kGradSuffix);
  }
  auto custom_name_prefix = custom_name.substr(0, pos);
  auto map_iter =
      paddle::OpMetaInfoMap::Instance().GetMap().find(custom_name_prefix);
  if (map_iter == paddle::OpMetaInfoMap::Instance().GetMap().end()) {
    PADDLE_THROW("The info of custom op : " + custom_name + " is not exists!");
  }
  const auto& vec_op_meta = map_iter->second;
  if (custom_name.find(kDoubleGradSuffix) != custom_name.npos) {
    return vec_op_meta[2];
  } else if (custom_name.find(kGradSuffix) != custom_name.npos) {
    return vec_op_meta[1];
  } else {
    return vec_op_meta[0];
  }
}

inline static bool HasGradOp(const std::string& fwd_pir_op_name) {
  auto custom_name = fwd_pir_op_name.substr(strlen(kCustomDialectPrefix));
  int pos = custom_name.length();

  if (custom_name[pos - 1] == '_') {
    // deal with inplace name
    custom_name = custom_name.substr(0, pos - 1);
  }

  pos = custom_name.length();
  if (custom_name.find(kDoubleGradSuffix) != custom_name.npos) {
    pos = custom_name.find(kDoubleGradSuffix);
  } else if (custom_name.find(kGradSuffix) != custom_name.npos) {
    pos = custom_name.find(kGradSuffix);
  }
  auto custom_name_prefix = custom_name.substr(0, pos);
  auto map_iter =
      paddle::OpMetaInfoMap::Instance().GetMap().find(custom_name_prefix);
  if (map_iter == paddle::OpMetaInfoMap::Instance().GetMap().end()) {
    PADDLE_THROW("The info of custom op : " + custom_name_prefix +
                 " is not exists!");
  }
  const auto& vec_op_meta = map_iter->second;
  if (custom_name.find(kDoubleGradSuffix) != custom_name.npos) {
    // custom op only support double grad, there will not have triple grad op
    return false;
  } else if (custom_name.find(kGradSuffix) != custom_name.npos) {
    // vec_op_meta.size() == 3 means the op has double grad op
    return vec_op_meta.size() > 2UL;
  } else {
    // vec_op_meta.size() == 2 or  vec_op_meta.size() == 3 means the op has grad
    // op
    return vec_op_meta.size() > 1UL;
  }
}
}  // namespace detail

static void CheckDefaultInferShapeDtype(
    paddle::InferShapeFunc infershape_func,
    paddle::InferDtypeFunc inferdtype_func,
    const paddle::OpMetaInfo& custom_op_meta) {
  if (infershape_func && inferdtype_func) {
    return;
  }
  auto& inplace_map = OpMetaInfoHelper::GetInplaceMap(custom_op_meta);
  if (inplace_map.empty()) {  // general case, assure single input and output
    PADDLE_ENFORCE_EQ(
        OpMetaInfoHelper::GetInputs(custom_op_meta).size(),
        1UL,
        common::errors::Unavailable(
            "Your custom operator contains multiple inputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the "
            "InferShapeFn/InferDtypeFn. "
            "At this time, the input shape/dtype will be directly set to "
            "the output shape/dtype.\n"
            "Please set the InferShapeFn/InferDtypeFn of custom "
            "operator by .SetInferShapeFn(PD_INFER_SHAPE(...)) / "
            ".SetInferDtypeFn(PD_INFER_DTYPE(...))"));
    PADDLE_ENFORCE_EQ(
        OpMetaInfoHelper::GetOutputs(custom_op_meta).size(),
        1UL,
        common::errors::Unavailable(
            "Your custom operator contains multiple outputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the "
            "InferShapeFn/InferDtypeFn. "
            "At this time, the input shape/dtype will be directly set to "
            "the output shape/dtype.\n"
            "Please set the InferShapeFn/InferDtypeFn of custom "
            "operator by .SetInferShapeFn(PD_INFER_SHAPE(...)) / "
            ".SetInferDtypeFn(PD_INFER_DTYPE(...))"));
  } else {  // inplace case
    PADDLE_ENFORCE_EQ(
        inplace_map.size(),
        OpMetaInfoHelper::GetOutputs(custom_op_meta).size(),
        common::errors::Unavailable(
            "Your custom operator uses `SetInplaceMap` without setting the "
            "InferShapeFn/InferDtypeFn. However, `Outputs` size = %d does not "
            "match the "
            "`InplaceMap` size = %d. Please check `SetInplaceMap` again or set "
            "the InferShapeFn/InferDtypeFn of custom operator by "
            ".SetInferShapeFn(PD_INFER_SHAPE(...)) / "
            ".SetInferDtypeFn(PD_INFER_DTYPE(...))",
            OpMetaInfoHelper::GetOutputs(custom_op_meta).size(),
            inplace_map.size()));
  }
}

static std::vector<std::vector<int64_t>> RunDefaultInferShape(
    const paddle::OpMetaInfo& custom_op_meta,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::unordered_map<std::string, int>& input_name2id_map,
    const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
    const std::unordered_map<std::string, int>& vec_input_name2id_map) {
  std::vector<std::vector<int64_t>> output_shapes;
  auto& inplace_reverse_map =
      OpMetaInfoHelper::GetInplaceReverseMap(custom_op_meta);
  // Op is grad op
  if (custom_op_meta.IsGradOp() || custom_op_meta.IsDoubleGradOp()) {
    bool is_double_grad = custom_op_meta.IsDoubleGradOp();
    const auto& bwd_outputs_name =
        paddle::OpMetaInfoHelper::GetOutputs(custom_op_meta);
    const auto& bwd_inputs_name =
        paddle::OpMetaInfoHelper::GetInputs(custom_op_meta);
    // 1. if forward input exists, gradient's shape is same with forward
    // input
    // default
    //    [Suitable for most situations]
    // 2. if forward input not exists, and only contains one grad input and
    // output,
    //    use grad input shape as grad output shape
    //    [Suitable for the situation that forward input is not used as
    //    backward input]
    for (auto& out_name : bwd_outputs_name) {
      auto bwd_input_name = detail::NoGrad(out_name, is_double_grad);
      if (detail::IsDuplicableVar(bwd_input_name)) {
        // Duplicable forward var must as backward input
        int input_index = vec_input_name2id_map.at(bwd_input_name);
        auto input_shape = vec_input_shapes[input_index];
        output_shapes.insert(
            output_shapes.end(), input_shape.begin(), input_shape.end());
      } else {
        if (std::find(bwd_inputs_name.begin(),
                      bwd_inputs_name.end(),
                      bwd_input_name) != bwd_inputs_name.end()) {
          int input_index = input_name2id_map.at(bwd_input_name);
          auto input_shape = input_shapes[input_index];
          if (input_shape.size() == 0) {
            // if optional tensor is None, we don't need to infer shape
            continue;
          }
          output_shapes.push_back(input_shape);
        } else {
          PADDLE_ENFORCE_EQ(
              bwd_inputs_name.size() == 1UL && bwd_outputs_name.size() == 1UL,
              true,
              common::errors::Unavailable(
                  "Custom grad operator infershape error. "
                  "If a custom grad operator contains only one input and "
                  "only one output, the input shape will be directly set "
                  "to the output shape. Otherwise, Please set the forward "
                  "input as the grad operator's input or set the "
                  "InferShapeFn of custom grad operator by "
                  ".SetInferShapeFn(PD_INFER_SHAPE(...))"));
          output_shapes.push_back(input_shapes[0]);
        }
      }
    }
    return output_shapes;
  }

  // Op is forward op
  if (inplace_reverse_map
          .empty()) {  // general case, assure single input and output
    VLOG(3) << "Custom Operator: Default InferShape - share ddim.";
    if (input_shapes.size() == 1) {
      output_shapes = input_shapes;
    } else if (vec_input_shapes.size() == 1) {
      output_shapes = vec_input_shapes[0];
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "We only allow a custom operator that contains only one input "
          "and only one output without setting the InferShapeFn. "));
    }
  } else {  // inplace case
    const auto& outputs = paddle::OpMetaInfoHelper::GetOutputs(custom_op_meta);
    for (auto& output : outputs) {
      auto input_name = inplace_reverse_map.at(output);
      if (paddle::framework::detail::IsDuplicableVar(output)) {
        int input_index = vec_input_name2id_map.at(input_name);
        auto input_shape = vec_input_shapes[input_index];
        output_shapes.insert(
            output_shapes.end(), input_shape.begin(), input_shape.end());
      } else {
        int input_index = input_name2id_map.at(input_name);
        auto input_shape = input_shapes[input_index];
        if (input_shape.size() == 0) {
          // if optional tensor is None, we don't need to infer shape
          continue;
        }
        output_shapes.push_back(input_shape);
      }
    }
  }
  return output_shapes;
}

static std::vector<DataType> RunDefaultInferDtype(
    const paddle::OpMetaInfo& custom_op_meta,
    const std::vector<DataType>& input_dtypes,
    const std::unordered_map<std::string, int>& input_name2id_map,
    const std::vector<std::vector<DataType>>& vec_input_dtypes,
    const std::unordered_map<std::string, int>& vec_input_name2id_map) {
  std::vector<DataType> output_dtypes;
  auto& inplace_reverse_map =
      OpMetaInfoHelper::GetInplaceReverseMap(custom_op_meta);
  // Op is grad op
  if (custom_op_meta.IsGradOp() || custom_op_meta.IsDoubleGradOp()) {
    bool is_double_grad = custom_op_meta.IsDoubleGradOp();
    const auto& bwd_outputs_name =
        paddle::OpMetaInfoHelper::GetOutputs(custom_op_meta);
    const auto& bwd_inputs_name =
        paddle::OpMetaInfoHelper::GetInputs(custom_op_meta);
    // The reason is same as RunDefaultInferShape
    for (auto& out_name : bwd_outputs_name) {
      auto bwd_input_name = detail::NoGrad(out_name, is_double_grad);
      if (detail::IsDuplicableVar(bwd_input_name)) {
        // Duplicable forward var must as backward input
        int input_index = vec_input_name2id_map.at(bwd_input_name);
        auto input_dtype = vec_input_dtypes[input_index];
        output_dtypes.insert(
            output_dtypes.end(), input_dtype.begin(), input_dtype.end());
      } else {
        if (std::find(bwd_inputs_name.begin(),
                      bwd_inputs_name.end(),
                      bwd_input_name) != bwd_inputs_name.end()) {
          int input_index = input_name2id_map.at(bwd_input_name);
          auto input_dtype = input_dtypes[input_index];
          if (input_dtype == DataType::UNDEFINED) {
            // if optional tensor is None, we don't need to infer dtype
            continue;
          }
          output_dtypes.push_back(input_dtype);
        } else {
          // If there is no corresponding input for the output, set float as
          // default type.
          output_dtypes.push_back(DataType::FLOAT32);
        }
      }
    }
    return output_dtypes;
  }

  if (inplace_reverse_map
          .empty()) {  // general case, assure single input and output
    VLOG(3) << "Custom Operator: Default InferDtype - share ddim.";
    if (input_dtypes.size() == 1) {
      output_dtypes = input_dtypes;
    } else if (vec_input_dtypes.size() == 1) {
      output_dtypes = vec_input_dtypes[0];
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "We only allow a custom operator that contains only one input "
          "and only one output without setting the InferDtypeFn. "));
    }
  } else {  // inplace case
    const auto& outputs = paddle::OpMetaInfoHelper::GetOutputs(custom_op_meta);
    for (auto& output : outputs) {
      auto input_name = inplace_reverse_map.at(output);
      if (paddle::framework::detail::IsDuplicableVar(output)) {
        int input_index = vec_input_name2id_map.at(input_name);
        auto input_dtype = vec_input_dtypes[input_index];
        output_dtypes.insert(
            output_dtypes.end(), input_dtype.begin(), input_dtype.end());
      } else {
        int input_index = input_name2id_map.at(input_name);
        auto input_dtype = input_dtypes[input_index];
        if (input_dtype == DataType::UNDEFINED) {
          // if optional tensor is None, we don't need to infer dtype
          continue;
        }
        output_dtypes.push_back(input_dtype);
      }
    }
  }
  return output_dtypes;
}

static std::vector<std::vector<int64_t>> RunInferShape(
    paddle::InferShapeFunc infershape_func,
    const paddle::OpMetaInfo& custom_op_meta,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::unordered_map<std::string, int>& input_name2id_map,
    const std::vector<std::vector<std::vector<int64_t>>>& vec_input_shapes,
    const std::unordered_map<std::string, int>& vec_input_name2id_map,
    const std::vector<paddle::any>& custom_attrs) {
  if (infershape_func) {
    std::vector<std::vector<int64_t>> infershape_result =
        infershape_func(input_shapes, vec_input_shapes, custom_attrs);
    std::vector<std::vector<int64_t>> complete_result;
    const auto& outputs = paddle::OpMetaInfoHelper::GetOutputs(custom_op_meta);
    const auto& inplace_reverse_map =
        paddle::OpMetaInfoHelper::GetInplaceReverseMap(custom_op_meta);

    // The real output shape result is ( infershape func result + inplace output
    // result), because the infershape doesn't create output shape that belongs
    // to inplace output.
    size_t infershape_result_index = 0;
    for (auto& out_name : outputs) {
      if (paddle::framework::detail::IsDuplicableVar(out_name)) {
        PADDLE_ENFORCE(
            inplace_reverse_map.find(out_name) != inplace_reverse_map.end(),
            common::errors::InvalidArgument(
                "Custom operator only supports `paddle::Vec(...)` inputs and "
                "cannot support `paddle::Vec(...)` output without setting "
                "InplaceMap. If you have to use `paddle::Vec(...)` output, "
                "please indicate it by setting InplaceMap manually."));
        auto in_name = inplace_reverse_map.at(out_name);
        if (custom_op_meta.IsGradOp() || custom_op_meta.IsDoubleGradOp()) {
          const auto& bwd_op_name =
              paddle::OpMetaInfoHelper::GetOpName(custom_op_meta);
          bool is_double_grad_op =
              (bwd_op_name.find(kDoubleGradSuffix) != bwd_op_name.npos) ? true
                                                                        : false;
          in_name =
              paddle::framework::detail::NoGrad(out_name, is_double_grad_op);
        }
        auto index = vec_input_name2id_map.at(in_name);
        const auto& vec_input_shape = vec_input_shapes[index];
        complete_result.insert(complete_result.end(),
                               vec_input_shape.begin(),
                               vec_input_shape.end());
      } else {
        if (inplace_reverse_map.find(out_name) != inplace_reverse_map.end()) {
          auto in_name = inplace_reverse_map.at(out_name);
          auto index = input_name2id_map.at(in_name);
          if (input_shapes[index].size() == 0) {
            // if optional tensor is None, we don't need to infer shape，
            continue;
          }
          complete_result.push_back(input_shapes[index]);
        } else {
          complete_result.push_back(infershape_result[infershape_result_index]);
          infershape_result_index++;
        }
      }
    }
    return complete_result;
  } else {
    return RunDefaultInferShape(custom_op_meta,
                                input_shapes,
                                input_name2id_map,
                                vec_input_shapes,
                                vec_input_name2id_map);
  }
}

static std::vector<DataType> RunInferDtype(
    paddle::InferDtypeFunc inferdtype_func,
    const paddle::OpMetaInfo& custom_op_meta,
    const std::vector<DataType>& input_dtypes,
    const std::unordered_map<std::string, int>& input_name2id_map,
    const std::vector<std::vector<DataType>>& vec_input_dtypes,
    const std::unordered_map<std::string, int>& vec_input_name2id_map,
    const std::vector<paddle::any>& custom_attrs) {
  if (inferdtype_func) {
    std::vector<DataType> complete_result;
    const auto& outputs = paddle::OpMetaInfoHelper::GetOutputs(custom_op_meta);
    const auto& inplace_reverse_map =
        paddle::OpMetaInfoHelper::GetInplaceReverseMap(custom_op_meta);
    std::vector<DataType> inferdtype_result =
        inferdtype_func(input_dtypes, vec_input_dtypes, custom_attrs);

    // The real output dtype result is ( infershape func dtype + inplace output
    // dtype), because the inferdtype doesn't create output dtype that belongs
    // to inplace output.
    size_t inferdtype_result_index = 0;
    for (auto& out_name : outputs) {
      if (paddle::framework::detail::IsDuplicableVar(out_name)) {
        PADDLE_ENFORCE(
            inplace_reverse_map.find(out_name) != inplace_reverse_map.end(),
            common::errors::InvalidArgument(
                "Custom operator only supports `paddle::Vec(...)` inputs and "
                "cannot support `paddle::Vec(...)` output without setting "
                "InplaceMap. If you have to use `paddle::Vec(...)` output, "
                "please indicate it by setting InplaceMap manually."));
        auto in_name = inplace_reverse_map.at(out_name);
        if (custom_op_meta.IsGradOp() || custom_op_meta.IsDoubleGradOp()) {
          const auto& bwd_op_name =
              paddle::OpMetaInfoHelper::GetOpName(custom_op_meta);
          bool is_double_grad_op =
              (bwd_op_name.find(kDoubleGradSuffix) != bwd_op_name.npos) ? true
                                                                        : false;
          in_name =
              paddle::framework::detail::NoGrad(out_name, is_double_grad_op);
        }
        auto index = vec_input_name2id_map.at(in_name);
        const auto& vec_input_dtype = vec_input_dtypes[index];
        complete_result.insert(complete_result.end(),
                               vec_input_dtype.begin(),
                               vec_input_dtype.end());
      } else {
        if (inplace_reverse_map.find(out_name) != inplace_reverse_map.end()) {
          auto in_name = inplace_reverse_map.at(out_name);
          auto index = input_name2id_map.at(in_name);
          if (input_dtypes[index] == DataType::UNDEFINED) {
            // if optional tensor is None, we don't need to infer dtype
            continue;
          }
          complete_result.push_back(input_dtypes[index]);
        } else {
          complete_result.push_back(inferdtype_result[inferdtype_result_index]);
          inferdtype_result_index++;
        }
      }
    }
    return complete_result;
  } else {
    return RunDefaultInferDtype(custom_op_meta,
                                input_dtypes,
                                input_name2id_map,
                                vec_input_dtypes,
                                vec_input_name2id_map);
  }
}

static phi::distributed::SpmdInfo RunInferSpmd(
    const paddle::OpMetaInfo& op_info,
    const std::string& op_type,
    const paddle::dialect::ProcessMeshAttribute& op_mesh,
    const std::vector<pir::Value>& argument_inputs,
    const std::vector<paddle::any>& custom_attrs) {  // NOLINT
#ifdef PADDLE_WITH_DISTRIBUTE
  auto& infer_spmd_func = paddle::OpMetaInfoHelper::GetInferSpmdFn(op_info);
  if (infer_spmd_func == nullptr) {
    // TODO(Q4): support replicated rule for custom op
    PADDLE_THROW(common::errors::Unavailable(
        "We only allow a custom operator with specific SPMD rule in auto "
        "parallel mode, please register a SPMD for [%s] Op first.",
        op_type));
  }

  std::vector<paddle::CustomSpmdInferTensorArg> dist_meta_tensors;
  dialect::CvtAllInputsToDist(argument_inputs, op_mesh);
  for (auto& value : argument_inputs) {
    // optional value
    if (!value || !value.type()) {
      phi::distributed::DistMetaTensor meta_tensor;
      dist_meta_tensors.emplace_back(std::move(meta_tensor));
      // single value
    } else if (auto dist_type =
                   value.type().dyn_cast<dialect::DistTypeInterface>()) {
      auto meta_tensor = dialect::CvtToDistMetaTensor(
          value.type().dyn_cast<dialect::DistDenseTensorType>());
      dist_meta_tensors.emplace_back(std::move(meta_tensor));
      // vector values
    } else if (auto vec_type = value.type().dyn_cast<pir::VectorType>()) {
      std::vector<phi::distributed::DistMetaTensor> meta_tensors;
      for (size_t idx = 0; idx < vec_type.size(); ++idx) {
        auto meta_tensor = dialect::CvtToDistMetaTensor(
            vec_type[idx].dyn_cast<dialect::DistDenseTensorType>());
        meta_tensors.emplace_back(std::move(meta_tensor));
      }
      dist_meta_tensors.emplace_back(std::move(meta_tensors));
    } else {
      std::ostringstream print_stream;
      print_stream << value.type();
      PADDLE_THROW(common::errors::Unavailable(
          "We only allow a custom operator with optional/single/vector inputs "
          "in auto parallel mode. %s",
          print_stream.str()));
    }
  }

  auto spmd_info_tmp = infer_spmd_func(dist_meta_tensors, custom_attrs);
  phi::distributed::SpmdInfo spmd_info;

  // NOTE not need to flatten input
  spmd_info.first = spmd_info_tmp.first;
  // for (auto& e : spmd_info_tmp.first) {
  //   if (paddle::holds_alternative<phi::distributed::TensorDistAttr>(e)) {
  //     spmd_info.first.push_back(
  //         std::move(PADDLE_GET(phi::distributed::TensorDistAttr, e)));
  //   } else {
  //     for (auto& ee :
  //          PADDLE_GET(std::vector<phi::distributed::TensorDistAttr>, e)) {
  //       spmd_info.first.push_back(std::move(ee));
  //     }
  //   }
  // }

  // flatten output
  for (auto& e : spmd_info_tmp.second) {
    if (paddle::holds_alternative<phi::distributed::TensorDistAttr>(e)) {
      spmd_info.second.push_back(
          std::move(PADDLE_GET(phi::distributed::TensorDistAttr, e)));
    } else {
      for (auto& ee :
           PADDLE_GET(std::vector<phi::distributed::TensorDistAttr>, e)) {
        spmd_info.second.push_back(std::move(ee));
      }
    }
  }

  return spmd_info;
#else
  PADDLE_THROW(common::errors::Unavailable(
      "The parsing of `RunInferSpmd` is not supported in the current "
      "PaddlePaddle, please recompile and installPaddlePaddle with the option "
      "of `WITH_DISTRIBUTE=ON`."));
#endif
}

}  // namespace framework
}  // namespace paddle
