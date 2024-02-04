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
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace framework {
constexpr char kCustomDialectPrefix[] = "custom_op.";  // NOLINT
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
      platform::errors::NotFound(
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

inline static const OpMetaInfo& GetGradOpInfoByFwdPirName(
    const std::string& pir_op_name) {
  auto custom_name = pir_op_name.substr(strlen(kCustomDialectPrefix));
  int pos = custom_name.length();

  if (custom_name[pos - 1] == '_') {
    // deal with inplace name
    custom_name = custom_name.substr(0, pos - 1);
  }

  pos = custom_name.length();
  if (custom_name.find("_grad_grad") != custom_name.npos) {
    pos = custom_name.find("_grad_grad") + 1;
  } else if (custom_name.find("_grad") != custom_name.npos) {
    pos = custom_name.find("_grad") + 1;
  }
  auto custom_name_prefix = custom_name.substr(0, pos);
  auto map_iter =
      paddle::OpMetaInfoMap::Instance().GetMap().find(custom_name_prefix);
  if (map_iter == paddle::OpMetaInfoMap::Instance().GetMap().end()) {
    PADDLE_THROW("The info of custom op : " + custom_name + " is not exists!");
  }
  const auto& vec_op_meta = map_iter->second;
  if (custom_name.find("_grad_grad") != custom_name.npos) {
    PADDLE_THROW("Custom op : " + custom_name_prefix +
                 " doesn't support triple grad.");
  } else if (custom_name.find("_grad") != custom_name.npos) {
    return vec_op_meta[2];
  } else {
    return vec_op_meta[1];
  }
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
  if (custom_name.find("_grad_grad") != custom_name.npos) {
    pos = custom_name.find("_grad_grad");
  } else if (custom_name.find("_grad") != custom_name.npos) {
    pos = custom_name.find("_grad");
  }
  auto custom_name_prefix = custom_name.substr(0, pos);
  auto map_iter =
      paddle::OpMetaInfoMap::Instance().GetMap().find(custom_name_prefix);
  if (map_iter == paddle::OpMetaInfoMap::Instance().GetMap().end()) {
    PADDLE_THROW("The info of custom op : " + custom_name + " is not exists!");
  }
  const auto& vec_op_meta = map_iter->second;
  if (custom_name.find("_grad_grad") != custom_name.npos) {
    return vec_op_meta[2];
  } else if (custom_name.find("_grad") != custom_name.npos) {
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
  if (custom_name.find("_grad_grad") != custom_name.npos) {
    pos = custom_name.find("_grad_grad");
  } else if (custom_name.find("_grad") != custom_name.npos) {
    pos = custom_name.find("_grad");
  }
  auto custom_name_prefix = custom_name.substr(0, pos);
  auto map_iter =
      paddle::OpMetaInfoMap::Instance().GetMap().find(custom_name_prefix);
  if (map_iter == paddle::OpMetaInfoMap::Instance().GetMap().end()) {
    PADDLE_THROW("The info of custom op : " + custom_name_prefix +
                 " is not exists!");
  }
  const auto& vec_op_meta = map_iter->second;
  if (custom_name.find("_grad_grad") != custom_name.npos) {
    // custom op only support double grad, there will not have triple grad op
    return false;
  } else if (custom_name.find("_grad") != custom_name.npos) {
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
        phi::errors::Unavailable(
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
        phi::errors::Unavailable(
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
        phi::errors::Unavailable(
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
  auto& inplace_map = OpMetaInfoHelper::GetInplaceMap(custom_op_meta);
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
          output_shapes.push_back(input_shape);
        } else {
          PADDLE_ENFORCE_EQ(
              bwd_inputs_name.size() == 1UL && bwd_outputs_name.size() == 1UL,
              true,
              paddle::platform::errors::Unavailable(
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
  if (inplace_map.empty()) {  // general case, assure single input and output
    VLOG(3) << "Custom Operator: Default InferShape - share ddim.";
    if (input_shapes.size() == 1) {
      output_shapes = input_shapes;
    } else if (vec_input_shapes.size() == 1) {
      output_shapes = vec_input_shapes[0];
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "We only allow a custom operator that contains only one input "
          "and only one output without setting the InferShapeFn. "));
    }
  } else {  // inplace case
    for (auto const& pair : inplace_map) {
      if (paddle::framework::detail::IsDuplicableVar(pair.second)) {
        int input_index = vec_input_name2id_map.at(pair.first);
        auto input_shape = vec_input_shapes[input_index];
        output_shapes.insert(
            output_shapes.end(), input_shape.begin(), input_shape.end());
      } else {
        int input_index = input_name2id_map.at(pair.first);
        auto input_shape = input_shapes[input_index];
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
  auto& inplace_map = OpMetaInfoHelper::GetInplaceMap(custom_op_meta);
  if (inplace_map.empty()) {  // general case, assure single input and output
    VLOG(3) << "Custom Operator: Default InferDtype - share ddim.";
    if (input_dtypes.size() == 1) {
      output_dtypes = input_dtypes;
    } else if (vec_input_dtypes.size() == 1) {
      output_dtypes = vec_input_dtypes[0];
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "We only allow a custom operator that contains only one input "
          "and only one output without setting the InferDtypeFn. "));
    }
  } else {  // inplace case
    for (auto const& pair : inplace_map) {
      if (paddle::framework::detail::IsDuplicableVar(pair.second)) {
        int input_index = vec_input_name2id_map.at(pair.first);
        auto input_dtype = vec_input_dtypes[input_index];
        output_dtypes.insert(
            output_dtypes.end(), input_dtype.begin(), input_dtype.end());
      } else {
        int input_index = input_name2id_map.at(pair.first);
        auto input_dtype = input_dtypes[input_index];
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
    return infershape_func(input_shapes, vec_input_shapes, custom_attrs);
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
    return inferdtype_func(input_dtypes, vec_input_dtypes, custom_attrs);
  } else {
    return RunDefaultInferDtype(custom_op_meta,
                                input_dtypes,
                                input_name2id_map,
                                vec_input_dtypes,
                                vec_input_name2id_map);
  }
}

}  // namespace framework
}  // namespace paddle
