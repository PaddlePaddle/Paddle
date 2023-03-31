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

#include "paddle/fluid/framework/custom_operator.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/api/all.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/utils/any.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif

#include "gflags/gflags.h"
#include "paddle/phi/api/include/operants_manager.h"
#include "paddle/phi/api/include/tensor_operants.h"

DECLARE_string(tensor_operants_mode);

namespace paddle {
namespace framework {

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

static std::vector<std::string> ParseAttrStr(const std::string& attr) {
  auto split_pos = attr.find_first_of(":");
  PADDLE_ENFORCE_NE(split_pos,
                    std::string::npos,
                    platform::errors::InvalidArgument(
                        "Invalid attribute string format. Attribute string "
                        "format is `<name>:<type>`."));

  std::vector<std::string> rlt;
  // 1. name
  rlt.emplace_back(string::trim_spaces(attr.substr(0, split_pos)));
  // 2. type
  rlt.emplace_back(string::trim_spaces(attr.substr(split_pos + 1)));

  VLOG(3) << "attr name: " << rlt[0] << ", attr type str: " << rlt[1];

  return rlt;
}

}  // namespace detail

////////////////// Kernel Define ////////////////////

// custom op kernel call function define
static void RunKernelFunc(
    const framework::ExecutionContext& ctx,
    const paddle::KernelFunc& func,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& attrs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  VLOG(3) << "Custom Operator: Start run KernelFunc.";
  // prepare CustomOpKernelContext
  paddle::CustomOpKernelContext kernel_ctx;
  for (auto& in_name : inputs) {
    VLOG(3) << "Custom Operator: input name - " << in_name;
    if (detail::IsDuplicableVar(in_name)) {  // inputs vector<Tensor>
      std::vector<paddle::Tensor> custom_vec_in;
      if (ctx.HasInputs(in_name)) {  // general vector<Tensor> inputs
        // return const std::vector<const phi::DenseTensor*>
        auto vec_x = ctx.MultiInput<phi::DenseTensor>(in_name);
        PADDLE_ENFORCE_NE(vec_x.empty(),
                          true,
                          platform::errors::NotFound(
                              "Input vector<tensor> (%s) is empty.", in_name));
        for (size_t i = 0; i < vec_x.size(); ++i) {
          auto* x = vec_x[i];
          PADDLE_ENFORCE_NOT_NULL(
              x,
              platform::errors::NotFound(
                  "The %d-th tensor in input vector<tensor> (%s) is nullptr.",
                  i,
                  in_name));
          PADDLE_ENFORCE_EQ(x->IsInitialized(),
                            true,
                            platform::errors::InvalidArgument(
                                "The %d-th tensor in input vector<tensor> (%s) "
                                "is not initialized.",
                                i,
                                in_name));
          paddle::Tensor custom_t;
          custom_t.set_impl(std::make_shared<phi::DenseTensor>(*x));
          custom_vec_in.emplace_back(custom_t);
        }
      } else {  // optional vector<Tensor> inputs.
        PADDLE_ENFORCE(
            detail::IsOptionalVar(in_name),
            phi::errors::NotFound("Your custom operator's KernelFunc cannot "
                                  "find input parameter `%s`",
                                  in_name));
        VLOG(3) << "Custom Operator: KernelFunc's vector input " << in_name
                << " is optional dtype with None input";
        // NOTE(HongyuJia): In dygraph mode, we can not distinguish Tensor and
        // vector<Tensor> when user inputs None, so dygraph mode appends one
        // un-initialized Tensor to CustomOpKernelContext. To be compatible with
        // dygraph mode, `custom_vec_in` also emplace_back one un-initialized
        // tensor here.
        custom_vec_in.emplace_back(paddle::Tensor());
      }
      kernel_ctx.EmplaceBackInputs(std::move(custom_vec_in));
    } else {                        // inputs Tensor
      if (ctx.HasInput(in_name)) {  // general Tensor inputs
        auto* x = ctx.Input<phi::DenseTensor>(in_name);
        PADDLE_ENFORCE_NOT_NULL(x,
                                platform::errors::NotFound(
                                    "Input tensor (%s) is nullptr.", in_name));
        PADDLE_ENFORCE_EQ(
            x->IsInitialized(),
            true,
            platform::errors::InvalidArgument(
                "Input tensor (%s) is not initialized.", in_name));
        paddle::Tensor custom_in;
        custom_in.set_impl(std::make_shared<phi::DenseTensor>(*x));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        if (custom_in.is_gpu_pinned()) {
          VLOG(3) << "Custom Operator: custom input is gpu pinned tensor";
          auto gpu_place = phi::GPUPlace(platform::GetCurrentDeviceId());
          auto custom_gpu_in = custom_in.copy_to(gpu_place, true);
          kernel_ctx.EmplaceBackInput(std::move(custom_gpu_in));
        } else {
          kernel_ctx.EmplaceBackInput(std::move(custom_in));
        }
#else
        kernel_ctx.EmplaceBackInput(std::move(custom_in));
#endif
      } else {  // optional Tensor inputs
        PADDLE_ENFORCE(
            detail::IsOptionalVar(in_name),
            phi::errors::NotFound("Your custom operator's KernelFunc cannot "
                                  "find input parameter `%s`",
                                  in_name));
        VLOG(3) << "Custom Operator: KernelFunc's input " << in_name
                << " is optional dtype with None input";
        kernel_ctx.EmplaceBackInput(std::move(paddle::Tensor()));
      }
    }
  }

  for (auto& attr_str : attrs) {
    auto attr_name_and_type = detail::ParseAttrStr(attr_str);
    auto attr_name = attr_name_and_type[0];
    auto attr_type_str = attr_name_and_type[1];
    if (attr_type_str == "bool") {
      kernel_ctx.EmplaceBackAttr(ctx.Attr<bool>(attr_name));
    } else if (attr_type_str == "int") {
      kernel_ctx.EmplaceBackAttr(ctx.Attr<int>(attr_name));
    } else if (attr_type_str == "float") {
      kernel_ctx.EmplaceBackAttr(ctx.Attr<float>(attr_name));
    } else if (attr_type_str == "int64_t") {
      kernel_ctx.EmplaceBackAttr(ctx.Attr<int64_t>(attr_name));
    } else if (attr_type_str == "std::string") {
      kernel_ctx.EmplaceBackAttr(ctx.Attr<std::string>(attr_name));
    } else if (attr_type_str == "std::vector<int>") {
      kernel_ctx.EmplaceBackAttr(ctx.Attr<std::vector<int>>(attr_name));
    } else if (attr_type_str == "std::vector<float>") {
      kernel_ctx.EmplaceBackAttr(ctx.Attr<std::vector<float>>(attr_name));
    } else if (attr_type_str == "std::vector<int64_t>") {
      kernel_ctx.EmplaceBackAttr(ctx.Attr<std::vector<int64_t>>(attr_name));
    } else if (attr_type_str == "std::vector<std::string>") {
      kernel_ctx.EmplaceBackAttr(ctx.Attr<std::vector<std::string>>(attr_name));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported `%s` type value as custom attribute now. "
          "Supported data types include `bool`, `int`, `float`, "
          "`int64_t`, `std::string`, `std::vector<int>`, "
          "`std::vector<float>`, `std::vector<int64_t>`, "
          "`std::vector<std::string>`, Please check whether "
          "the attribute data type and data type string are matched.",
          attr_type_str));
    }
  }

  VLOG(3) << "Custom Operator: push outputs into CustomOpKernelContext.";
  // cache the target tensor pointers
  std::vector<phi::DenseTensor*> true_out_ptrs;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto out_name = outputs[i];
    if (detail::IsDuplicableVar(
            out_name)) {  // general/inplace vector<Tensor> outputs
      PADDLE_ENFORCE(
          !inplace_map.empty() || (i == 0UL && outputs.size() == 1UL),
          phi::errors::PreconditionNotMet(
              "If custom operator's outputs contains `paddle::Vec()` type "
              "without setting InplaceMap, it only can hold one output."));
      auto vec_out = ctx.MultiOutput<phi::DenseTensor>(out_name);
      // handle inplace optional outputs = None case
      if (vec_out.empty()) {
        PADDLE_ENFORCE(
            detail::IsOptionalVar(out_name) && !inplace_map.empty(),
            phi::errors::InvalidArgument(
                "Custom operator couldn't find custom output for name %s. If "
                "you "
                "are using inplace optional inputs & outputs, please check "
                "your "
                "InplaceMap and `Outputs` again and make sure %s is wrapped by "
                "`paddle::Optional`",
                out_name,
                out_name));
        VLOG(3) << "Custom Operator: InferDtype - inplace optional outputs : "
                << out_name << " is None.";
        true_out_ptrs.emplace_back(nullptr);
        kernel_ctx.EmplaceBackOutput(std::move(paddle::Tensor()));
        continue;
      }
      // general/inplace vector<Tensor> outputs
      std::vector<paddle::Tensor> custom_vec_out;
      for (size_t j = 0; j < vec_out.size(); ++j) {
        auto* out = vec_out[j];
        PADDLE_ENFORCE_NOT_NULL(
            out,
            platform::errors::NotFound(
                "The %d-th tensor in output vector<tensor> (%s) is nullptr.",
                j,
                out_name));
        true_out_ptrs.emplace_back(out);
        paddle::Tensor custom_t;
        // here only can copy the output tensor into context
        custom_t.set_impl(std::make_shared<phi::DenseTensor>(*out));
        custom_vec_out.emplace_back(custom_t);
      }
      kernel_ctx.EmplaceBackOutputs(std::move(custom_vec_out));
    } else {
      // handle inplace optional outputs = None case
      if (!ctx.HasOutput(out_name)) {
        PADDLE_ENFORCE(
            detail::IsOptionalVar(out_name) && !inplace_map.empty(),
            phi::errors::InvalidArgument(
                "Custom operator couldn't find custom output for name %s. If "
                "you "
                "are using inplace optional inputs & outputs, please check "
                "your "
                "InplaceMap and `Outputs` again and make sure %s is wrapped by "
                "`paddle::Optional`",
                out_name,
                out_name));
        VLOG(3) << "Custom Operator: InferDtype - inplace optional outputs : "
                << out_name << " is None.";
        true_out_ptrs.emplace_back(nullptr);
        kernel_ctx.EmplaceBackOutput(std::move(paddle::Tensor()));
        continue;
      }
      // general/inplace Tensor outputs
      auto* out = ctx.Output<phi::DenseTensor>(out_name);
      PADDLE_ENFORCE_NOT_NULL(out,
                              platform::errors::NotFound(
                                  "Output tensor (%s) is nullptr.", out_name));
      true_out_ptrs.emplace_back(out);
      paddle::Tensor custom_out;
      // here only can copy the output tensor into context
      custom_out.set_impl(std::make_shared<phi::DenseTensor>(*out));
      kernel_ctx.EmplaceBackOutput(std::move(custom_out));
    }
  }

  try {
    VLOG(3) << "Custom Operator: Run ComputeFunc.";

    FLAGS_tensor_operants_mode = "phi";
    if (paddle::OperantsManager::Instance().phi_operants.get() == nullptr) {
      paddle::OperantsManager::Instance().phi_operants.reset(
          new paddle::operants::PhiTensorOperants());
      VLOG(4) << "Initialize phi tensor operants successfully";
    }

    // handle inplace map
    kernel_ctx.MapPlainOutputs(inputs, outputs, inplace_map);
    func(&kernel_ctx);
    kernel_ctx.AssignInplaceOutputs();

    // sync output tensor data into original output
    auto* calc_outs = kernel_ctx.AllMutableOutput();
    PADDLE_ENFORCE_EQ(
        true_out_ptrs.size(),
        calc_outs->size(),
        platform::errors::InvalidArgument(
            "The number of element in custom operator outputs is wrong, "
            "expected contains %d Tensors, but actually contains %d "
            "Tensors.",
            true_out_ptrs.size(),
            calc_outs->size()));
    for (size_t i = 0; i < true_out_ptrs.size(); ++i) {
      auto* true_out = true_out_ptrs.at(i);
      // handle optional inplace outputs = None case
      if (true_out == nullptr && !calc_outs->at(i).defined()) {
        continue;
      }
      PADDLE_ENFORCE(
          true_out != nullptr && calc_outs->at(i).defined(),
          platform::errors::InvalidArgument(
              "The returned Tensor is not defined in the KernelFn or custom "
              "operator passes wrong output in static mode."));
      auto calc_out =
          std::dynamic_pointer_cast<phi::DenseTensor>(calc_outs->at(i).impl());
      // assign meta info
      auto* true_out_meta = phi::DenseTensorUtils::GetMutableMeta(true_out);
      true_out_meta->dims = calc_out->dims();
      true_out_meta->dtype = calc_out->dtype();
      true_out_meta->layout = calc_out->layout();
      true_out_meta->offset = calc_out->offset();
      // lod no need to be reset
      // reset holder if needed
      if (true_out->Holder() != calc_out->Holder()) {
        true_out->ResetHolder(calc_out->Holder());
      }
    }
  } catch (platform::EnforceNotMet& exception) {
    throw std::move(exception);
  } catch (std::exception& ex) {
    PADDLE_THROW(platform::errors::External("%s", ex.what()));
  } catch (...) {
    PADDLE_THROW(platform::errors::Fatal(
        "Custom operator raises an unknown exception in runtime."));
  }
}

static void RunDefaultInferShapeFunc(
    framework::InferShapeContext* ctx,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  if (inplace_map.empty()) {  // general case, assure single input and output
    PADDLE_ENFORCE_EQ(
        inputs.size(),
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
        outputs.size(),
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
    ctx->ShareDim(inputs[0], outputs[0]);
  } else {  // inplace case
    PADDLE_ENFORCE_EQ(
        inplace_map.size(),
        outputs.size(),
        phi::errors::Unavailable(
            "Your custom operator uses `SetInplaceMap` without setting the "
            "InferShapeFn. However, `Outputs` size = %d does not match the "
            "`InplaceMap` size = %d. Please check `SetInplaceMap` again or set "
            "the InferShapeFn of custom operator by "
            "`.SetInferShapeFn(PD_INFER_SHAPE(...)`)",
            outputs.size(),
            inplace_map.size()));
    for (auto const& pair : inplace_map) {
      if (detail::IsDuplicableVar(pair.first)) {
        // make sure ctx has valid inplace optional outputs
        if (!ctx->HasOutputs(pair.second)) {
          PADDLE_ENFORCE(
              detail::IsOptionalVar(pair.second),
              phi::errors::InvalidArgument(
                  "Custom operator couldn't find custom output name for %s. If "
                  "you are using inplace optional inputs & outputs, please "
                  "check "
                  "your InplaceMap and `Outputs` again and make sure %s is "
                  "wrapped by `paddle::Optional`",
                  pair.second,
                  pair.second));
          VLOG(3) << "Custom Operator: InferDtype - inplace optional outputs : "
                  << pair.second << " is None.";
        } else {
          ctx->SetOutputsDim(pair.second, ctx->GetInputsDim(pair.first));
        }
      } else {
        // make sure ctx has valid inplace optional outputs
        if (!ctx->HasOutput(pair.second)) {
          PADDLE_ENFORCE(
              detail::IsOptionalVar(pair.second),
              phi::errors::InvalidArgument(
                  "Custom operator couldn't find custom output name for %s. If "
                  "you are using inplace optional inputs & outputs, please "
                  "check "
                  "your InplaceMap and `Outputs` again and make sure %s is "
                  "wrapped by `paddle::Optional`",
                  pair.second,
                  pair.second));
          VLOG(3) << "Custom Operator: InferDtype - inplace optional outputs : "
                  << pair.second << " is None.";
        } else {
          ctx->ShareDim(pair.first, pair.second);
        }
      }
    }
  }
}

static void RunInferShapeFunc(
    framework::InferShapeContext* ctx,
    const paddle::InferShapeFunc& func,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& attrs,
    const std::unordered_map<std::string, std::string>& inplace_map,
    const std::unordered_map<std::string, std::string>& inplace_reverse_map) {
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<std::vector<int64_t>>> vec_input_shapes;

  VLOG(3) << "Custom Operator: InferShape - get input ddim.";
  for (auto& in_name : inputs) {
    if (detail::IsDuplicableVar(in_name)) {
      std::vector<std::vector<int64_t>> vec_shape;
      if (ctx->HasInputs(in_name)) {  // general inputs
        auto vec_ddim = ctx->GetInputsDim(in_name);
        vec_shape.reserve(vec_ddim.size());
        std::transform(vec_ddim.begin(),
                       vec_ddim.end(),
                       std::back_inserter(vec_shape),
                       [&](const DDim& ddim) -> std::vector<int64_t> {
                         return phi::vectorize(ddim);
                       });

      } else {  // optional inputs, `vec_shape` is empty
        PADDLE_ENFORCE(
            detail::IsOptionalVar(in_name),
            phi::errors::NotFound("Your custom operator's InferShapeFunc "
                                  "cannot find input parameter `%s`",
                                  in_name));
        VLOG(3) << "Custom Operator: InferShapeFunc's vector input " << in_name
                << " is optional dtype with None input";
      }
      vec_input_shapes.emplace_back(vec_shape);
    } else {
      if (ctx->HasInput(in_name)) {  // general inputs
        auto ddim = ctx->GetInputDim(in_name);
        input_shapes.emplace_back(phi::vectorize(ddim));
      } else {  // optional inputs
        PADDLE_ENFORCE(
            detail::IsOptionalVar(in_name),
            phi::errors::NotFound("Your custom operator's InferShapeFunc "
                                  "cannot find input parameter `%s`",
                                  in_name));
        input_shapes.emplace_back(std::vector<int64_t>());
        VLOG(3) << "Custom Operator: InferShapeFunc's input " << in_name
                << " is optional dtype with None input";
      }
    }
  }

  std::vector<paddle::any> custom_attrs;
  for (auto& attr_str : attrs) {
    auto attr_name_and_type = detail::ParseAttrStr(attr_str);
    auto attr_name = attr_name_and_type[0];
    auto attr_type_str = attr_name_and_type[1];
    if (attr_type_str == "bool") {
      custom_attrs.emplace_back(ctx->Attrs().Get<bool>(attr_name));
    } else if (attr_type_str == "int") {
      custom_attrs.emplace_back(ctx->Attrs().Get<int>(attr_name));
    } else if (attr_type_str == "float") {
      custom_attrs.emplace_back(ctx->Attrs().Get<float>(attr_name));
    } else if (attr_type_str == "int64_t") {
      custom_attrs.emplace_back(ctx->Attrs().Get<int64_t>(attr_name));
    } else if (attr_type_str == "std::string") {
      custom_attrs.emplace_back(ctx->Attrs().Get<std::string>(attr_name));
    } else if (attr_type_str == "std::vector<int>") {
      custom_attrs.emplace_back(ctx->Attrs().Get<std::vector<int>>(attr_name));
    } else if (attr_type_str == "std::vector<float>") {
      custom_attrs.emplace_back(
          ctx->Attrs().Get<std::vector<float>>(attr_name));
    } else if (attr_type_str == "std::vector<int64_t>") {
      // NOTE(chenweihang): InferShape can't support std::vector<int64_t>
      // attr type, because the input type is std::vector<int64_t>, only
      // can use one rule to parse std::vector<int64_t> parameter
      continue;
    } else if (attr_type_str == "std::vector<std::string>") {
      custom_attrs.emplace_back(
          ctx->Attrs().Get<std::vector<std::string>>(attr_name));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported `%s` type value as custom attribute now. "
          "Supported data types include `bool`, `int`, `float`, "
          "`int64_t`, `std::string`, `std::vector<int>`, "
          "`std::vector<float>`, `std::vector<std::string>`, "
          "Please check whether the attribute data type and "
          "data type string are matched.",
          attr_type_str));
    }
  }

  VLOG(3) << "Custom Operator: InferShape - calc output ddim.";
  auto output_shapes = func(input_shapes, vec_input_shapes, custom_attrs);
  if (inplace_map.empty()) {
    PADDLE_ENFORCE_EQ(outputs.size(),
                      output_shapes.size(),
                      phi::errors::InvalidArgument(
                          "Your custom operator has set the InferShapeFn. "
                          "However, `Outputs` size = %d does not match the "
                          "returned vector size of InferShapeFn = %d. Please "
                          "check InferShapeFn again.",
                          outputs.size(),
                          output_shapes.size()));
  } else {
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        output_shapes.size() + inplace_map.size(),
        phi::errors::InvalidArgument(
            "Your custom operator uses `SetInplaceMap` and sets the "
            "InferShapeFn. However, `Outputs` size = %d does not match the "
            "`InplaceMap size + InferShapeFn output size` = %d. Please check "
            "InplaceMap and InferShapeFn again",
            outputs.size(),
            output_shapes.size() + inplace_map.size()));
  }

  VLOG(3)
      << "Custom Operator: InferShape - set output ddim: inplace_map.size() = "
      << inplace_map.size()
      << ", output_shapes.size() = " << output_shapes.size();
  size_t output_shape_idx = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto out_name = outputs[i];
    if (detail::IsDuplicableVar(out_name)) {
      PADDLE_ENFORCE(
          inplace_reverse_map.find(out_name) != inplace_reverse_map.end(),
          phi::errors::InvalidArgument(
              "Custom operator only supports `paddle::Vec(...)` inputs and "
              "cannot support `paddle::Vec(...)` output without setting "
              "InplaceMap. If you have to use `paddle::Vec(...)` output, "
              "please indicate it by setting InplaceMap manully."));
      // make sure ctx has valid inplace optional outputs
      if (ctx->HasOutputs(out_name)) {
        auto in_name = inplace_reverse_map.at(out_name);
        ctx->SetOutputsDim(out_name, ctx->GetInputsDim(in_name));
      } else {
        PADDLE_ENFORCE(
            detail::IsOptionalVar(out_name),
            phi::errors::InvalidArgument(
                "Custom operator couldn't find custom output name for %s. If "
                "you are using inplace optional inputs & outputs, please check "
                "your InplaceMap and `Outputs` again and make sure %s is "
                "wrapped by `paddle::Optional`",
                out_name,
                out_name));
        VLOG(3) << "Custom Operator: InferDtype - inplace optional outputs : "
                << out_name << " is None.";
      }
    } else {
      if (inplace_reverse_map.find(out_name) != inplace_reverse_map.end()) {
        // make sure ctx has valid inplace optional outputs
        if (ctx->HasOutput(out_name)) {
          // Share dims between inplace inputs and outputs
          ctx->ShareDim(inplace_reverse_map.at(out_name), out_name);
        } else {
          PADDLE_ENFORCE(
              detail::IsOptionalVar(out_name),
              phi::errors::InvalidArgument(
                  "Custom operator couldn't find custom output name for %s. If "
                  "you are using inplace optional inputs & outputs, please "
                  "check your InplaceMap and `Outputs` again and make sure %s "
                  "is wrapped by `paddle::Optional`",
                  out_name,
                  out_name));
          VLOG(3) << "Custom Operator: InferDtype - inplace optional outputs : "
                  << out_name << " is None.";
        }
      } else {
        // Set output dims by the output of InferShapeFn
        ctx->SetOutputDim(out_name,
                          phi::make_ddim(output_shapes[output_shape_idx++]));
      }
    }
  }
}

static void RunDefaultInferDtypeFunc(
    framework::InferVarTypeContext* ctx,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map) {
  if (inplace_map.empty()) {  // general case, assure single input and output
    PADDLE_ENFORCE_EQ(
        inputs.size(),
        1UL,
        platform::errors::Unavailable(
            "Your custom operator contains multiple inputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferDtypeFn. "
            "At this time, the input dtype will be directly set to "
            "the output dtype.\n"
            "Please set the InferDtypeFn of custom "
            "operator by `.SetInferDtypeFn(PD_INFER_DTYPE(...))`"));
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        1UL,
        platform::errors::Unavailable(
            "Your custom operator contains multiple outputs. "
            "We only allow a custom operator that contains only one input "
            "and only one output without setting the InferDtypeFn. "
            "At this time, the input dtype will be directly set to "
            "the output dtype.\n"
            "Please set the InferDtypeFn of custom "
            "operator by `.SetInferDtypeFn(PD_INFER_DTYPE(...))`"));

    VLOG(3) << "Custom Operator: InferDtype - share dtype.";
    auto dtype = ctx->GetInputDataType(inputs[0]);
    ctx->SetOutputDataType(outputs[0], dtype);
  } else {  // inplace case
    PADDLE_ENFORCE_EQ(
        inplace_map.size(),
        outputs.size(),
        phi::errors::Unavailable(
            "Your custom operator uses `SetInplaceMap` without setting the "
            "InferDtypeFn. However, `Outputs` size = %d does not match the "
            "`InplaceMap` size = %d. Please check `SetInplaceMap` again or set "
            "the InferDtypeFn of custom operator by "
            "`.SetInferDtypeFn(PD_INFER_DTYPE(...))`",
            outputs.size(),
            inplace_map.size()));
    for (auto const& pair : inplace_map) {
      VLOG(3) << "Custom Operator: InferDtype - inplace dtype: " << pair.first
              << "->" << pair.second;
      // make sure ctx has valid inplace optional outputs
      if (!ctx->HasOutput(pair.second)) {
        PADDLE_ENFORCE(
            detail::IsOptionalVar(pair.second),
            phi::errors::InvalidArgument(
                "Custom operator couldn't find custom output name for %s. If "
                "you are using inplace optional inputs & outputs, please check "
                "your InplaceMap and `Outputs` again and make sure %s is "
                "wrapped by `paddle::Optional`",
                pair.second,
                pair.second));
        VLOG(3) << "Custom Operator: InferDtype - inplace optional outputs : "
                << pair.second << " is None.";
        continue;
      }
      if (detail::IsDuplicableVar(pair.first)) {
        size_t size = ctx->InputSize(pair.first);
        for (size_t i = 0; i < size; ++i) {
          auto dtype = ctx->GetInputDataType(pair.first, i);
          ctx->SetOutputDataType(pair.second, dtype, i);
        }
      } else {
        auto dtype = ctx->GetInputDataType(pair.first);
        ctx->SetOutputDataType(pair.second, dtype);
      }
    }
  }
}

static void RunInferDtypeFunc(
    framework::InferVarTypeContext* ctx,
    const paddle::InferDtypeFunc& func,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::unordered_map<std::string, std::string>& inplace_map,
    const std::unordered_map<std::string, std::string>& inplace_reverse_map) {
  std::vector<DataType> input_dtypes;
  std::vector<std::vector<DataType>> vec_input_dtypes;

  VLOG(3) << "Custom Operator: InferDtype - get input dtype.";
  for (auto& in_name : inputs) {
    if (detail::IsDuplicableVar(in_name)) {
      std::vector<DataType> vec_custom_dtype;
      if (ctx->HasInput(in_name)) {  // general inputs
        for (size_t i = 0; i < ctx->InputSize(in_name); ++i) {
          auto dtype = ctx->GetInputDataType(in_name, i);
          vec_custom_dtype.emplace_back(
              paddle::framework::TransToPhiDataType(dtype));
        }
      } else {  // optional inputs, `vec_custom_dtype` is empty
        PADDLE_ENFORCE(
            detail::IsOptionalVar(in_name),
            phi::errors::NotFound("Your custom operator's InferDtypeFn "
                                  "cannot find input parameter `%s`",
                                  in_name));
        VLOG(3) << "Custom Operator: InferDtypeFn's vector input " << in_name
                << " is optional dtype with None input";
      }
      vec_input_dtypes.emplace_back(vec_custom_dtype);
    } else {
      if (ctx->HasInput(in_name)) {  // general inputs
        auto dtype = ctx->GetInputDataType(in_name);
        input_dtypes.emplace_back(paddle::framework::TransToPhiDataType(dtype));
      } else {  // optional inputs
        PADDLE_ENFORCE(
            detail::IsOptionalVar(in_name),
            phi::errors::NotFound("Your custom operator's InferDtypeFn "
                                  "cannot find input parameter `%s`",
                                  in_name));
        input_dtypes.emplace_back(DataType::UNDEFINED);
        VLOG(3) << "Custom Operator: InferDtypeFn's input " << in_name
                << " is optional dtype with None input";
      }
    }
  }

  VLOG(3) << "Custom Operator: InferDtype - infer output dtype.";
  auto output_dtypes = func(input_dtypes, vec_input_dtypes);
  if (inplace_map.empty()) {
    PADDLE_ENFORCE_EQ(outputs.size(),
                      output_dtypes.size(),
                      phi::errors::InvalidArgument(
                          "Your custom operator has set the InferDtypeFn. "
                          "However, `Outputs` size = %d does not match the "
                          "returned vector size of InferDtypeFn = %d. Please "
                          "check InferDtypeFn again.",
                          outputs.size(),
                          output_dtypes.size()));
  } else {
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        output_dtypes.size() + inplace_map.size(),
        phi::errors::InvalidArgument(
            "Your custom operator uses `SetInplaceMap` and sets the "
            "InferDtypeFn. However, `Outputs` size = %d does not match the "
            "`InplaceMap size + InferDtypeFn output size` = %d. Please check "
            "InplaceMap and InferDtypeFn again",
            outputs.size(),
            output_dtypes.size() + inplace_map.size()));
  }

  VLOG(3)
      << "Custom Operator: InferDtype - set output dtype: inplace_map.size() = "
      << inplace_map.size()
      << ", output_dtypes.size() = " << output_dtypes.size();
  size_t output_dtype_idx = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto out_name = outputs[i];
    if (detail::IsDuplicableVar(out_name)) {
      PADDLE_ENFORCE(
          inplace_reverse_map.find(out_name) != inplace_reverse_map.end(),
          phi::errors::InvalidArgument(
              "Custom operator only supports `paddle::Vec(...)` inputs and "
              "cannot support `paddle::Vec(...)` output without setting "
              "InplaceMap. If you have to use `paddle::Vec(...)` output, "
              "please indicate it by setting InplaceMap manully."));
      auto in_name = inplace_reverse_map.at(out_name);
      // make sure ctx has valid inplace optional outputs
      if (ctx->HasOutput(out_name)) {
        ctx->SetOutputDataTypes(out_name, ctx->GetInputDataTypes(in_name));
      } else {
        PADDLE_ENFORCE(
            detail::IsOptionalVar(out_name),
            phi::errors::InvalidArgument(
                "Custom operator couldn't find custom output name for %s. If "
                "you are using inplace optional inputs & outputs, please check "
                "your InplaceMap and `Outputs` again and make sure %s is "
                "wrapped by `paddle::Optional`",
                out_name,
                out_name));
        VLOG(3) << "Custom Operator: InferDtype - inplace optional outputs : "
                << out_name << " is None.";
      }
    } else {
      if (inplace_reverse_map.find(out_name) != inplace_reverse_map.end()) {
        // make sure ctx has valid inplace optional outputs
        if (ctx->HasOutput(out_name)) {
          auto in_name = inplace_reverse_map.at(out_name);
          // Share dtype between inplace inputs and outputs
          ctx->SetOutputDataType(out_name, ctx->GetInputDataType(in_name));
        } else {
          PADDLE_ENFORCE(
              out_name.find(paddle::kOptionalSuffix) != std::string::npos,
              phi::errors::InvalidArgument(
                  "Custom operator couldn't find custom output name for %s. If "
                  "you are using inplace optional inputs & outputs, please "
                  "check your InplaceMap and `Outputs` again and make sure %s "
                  "is wrapped by `paddle::Optional`",
                  out_name,
                  out_name));
          VLOG(3) << "Custom Operator: InferDtype - inplace optional outputs : "
                  << out_name << " is None.";
        }
      } else {
        // Set output dtype by the output of InferDtypeFn
        ctx->SetOutputDataType(out_name,
                               paddle::framework::TransToProtoVarType(
                                   output_dtypes[output_dtype_idx++]));
      }
    }
  }
}

//////////////////// Operator Define /////////////////

class CustomOperator : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  // Dummy infershape
  // Because it is a pure virtual function, it must be implemented
  void InferShape(framework::InferShapeContext* ctx) const override {
    VLOG(3) << "Custom Operator: Dummy infer shape of custom operator.";
  }

  /**
   * NOTE: [Skip the Kernel Selection]
   * Custom Op only registers one Op kernel on each device, so that the
   * data type selection and promotion that depends on GetExpectedKernelType,
   * as well as the adaptation of various other special situations,
   * need users to implement, to avoid users needs to implement
   * GetExpectedKernelType function when expanding other cases.
   * The RAW type is used here as the data type, indicating that
   * it can only be determined at runtime.
   */
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(ctx.GetPlace());
  }

  /**
   * NOTE: [Skip Input Variable Cast for DataType]
   * Because the kernel data type is RAW, we should skip the cast for
   * data type difference when PrepareData.
   */
  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    return phi::KernelKey(phi::Backend::ALL_BACKEND,
                          tensor.layout(),
                          expected_kernel_type.dtype());
  }
};

class CustomOpMaker : public OpProtoAndCheckerMaker {
 public:
  explicit CustomOpMaker(const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs,
                         const std::vector<std::string>& attrs)
      : inputs_(inputs), outputs_(outputs), attrs_(attrs) {}

  void Make() override {
    for (auto& in_name : inputs_) {
      auto input_var_builder =
          AddInput(in_name, "The input " + in_name + "of Custom operator.");
      if (detail::IsDuplicableVar(in_name)) {
        input_var_builder.AsDuplicable();
      }
      if (detail::IsOptionalVar(in_name)) {
        input_var_builder.AsDispensable();
      }
    }
    for (auto& out_name : outputs_) {
      auto output_var_builder =
          AddOutput(out_name, "The output " + out_name + "of Custom Operator.");
      if (detail::IsDuplicableVar(out_name)) {
        output_var_builder.AsDuplicable();
      }
      if (detail::IsOptionalVar(out_name)) {
        output_var_builder.AsDispensable();
      }
    }
    for (auto& attr : attrs_) {
      auto attr_name_and_type = detail::ParseAttrStr(attr);
      auto attr_name = attr_name_and_type[0];
      auto attr_type_str = attr_name_and_type[1];
      if (attr_type_str == "bool") {
        AddAttr<bool>(attr_name, "custom operator bool attribute.")
            .SetDefault(false);
      } else if (attr_type_str == "int") {
        AddAttr<int>(attr_name, "custom operator int attribute.").SetDefault(1);
      } else if (attr_type_str == "float") {
        AddAttr<float>(attr_name, "custom operator float attribute.")
            .SetDefault(1.0f);
      } else if (attr_type_str == "int64_t") {
        AddAttr<int64_t>(attr_name, "custom operator int64_t attribute.")
            .SetDefault(1);
      } else if (attr_type_str == "std::string") {
        AddAttr<std::string>(attr_name, "custom operator int attribute.")
            .SetDefault("");
      } else if (attr_type_str == "std::vector<int>") {
        AddAttr<std::vector<int>>(attr_name,
                                  "custom operator std::vector<int> attribute.")
            .SetDefault({});
      } else if (attr_type_str == "std::vector<float>") {
        AddAttr<std::vector<float>>(
            attr_name, "custom operator std::vector<float> attribute.")
            .SetDefault({});
      } else if (attr_type_str == "std::vector<int64_t>") {
        AddAttr<std::vector<int64_t>>(
            attr_name, "custom operator std::vector<int64_t> attribute.")
            .SetDefault({});
      } else if (attr_type_str == "std::vector<std::string>") {
        AddAttr<std::vector<std::string>>(
            attr_name, "custom operator std::vector<std::string> attribute.")
            .SetDefault({});
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported `%s` type value as custom attribute now. "
            "Supported data types include `bool`, `int`, `float`, "
            "`int64_t`, `std::string`, `std::vector<int>`, "
            "`std::vector<float>`, `std::vector<int64_t>`, "
            "`std::vector<std::string>`, Please check whether "
            "the attribute data type and data type string are matched.",
            attr_type_str));
      }
    }
    AddComment(R"DOC(
Custom Operator.

According to the phi::DenseTensor operation function implemented by the user
independently of the framework, it is encapsulated into a framework
operator to adapt to various execution scenarios such as dynamic graph
mode, static graph mode, and inference mode.

)DOC");
  }

 private:
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::vector<std::string> attrs_;
};

template <typename T>
class CustomGradOpMaker;

template <>
class CustomGradOpMaker<OpDesc> : public SingleGradOpMaker<OpDesc> {
 public:
  explicit CustomGradOpMaker(
      const OpDesc& fwd_op,
      const std::unordered_set<std::string>& no_grad_set,
      std::unordered_map<std::string, std::string>* grad_to_var,
      const std::vector<BlockDesc*>& grad_block,
      const std::string& name,
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs,
      bool is_double_grad)
      : SingleGradOpMaker<OpDesc>(fwd_op, no_grad_set, grad_to_var, grad_block),
        name_(name),
        inputs_(inputs),
        outputs_(outputs),
        is_double_grad_(is_double_grad) {}

 protected:
  void Apply(GradOpPtr<OpDesc> grad_op) const override {
    grad_op->SetType(name_);

    auto fwd_op_inputs = this->InputNames();
    auto fwd_op_outputs = this->OutputNames();

    for (auto& in_name : inputs_) {
      VLOG(3) << "Custom Operator: GradOpDescMaker - input: " << in_name;
      if (!detail::IsGradVar(in_name, is_double_grad_)) {
        if (detail::IsMemberOf(fwd_op_inputs, in_name)) {
          grad_op->SetInput(in_name, this->Input(in_name));
        } else if (detail::IsMemberOf(fwd_op_outputs, in_name)) {
          grad_op->SetInput(in_name, this->Output(in_name));
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "The input tensor name `%s` is invalid, expected it is the input "
              "or output of forward operator.",
              in_name));
        }
      } else {
        if (this->HasOutput(detail::NoGrad(in_name))) {
          grad_op->SetInput(in_name, this->OutputGrad(detail::NoGrad(in_name)));
        } else {
          // Maybe visit here! handle inplace optional case
          PADDLE_ENFORCE(
              in_name.find(paddle::kOptionalSuffix) != std::string::npos,
              phi::errors::InvalidArgument(
                  "Custom operator couldn't find grad operator input name for "
                  "%s. If you are using inplace optional inputs & outputs, "
                  "please check your InplaceMap and `Outputs` again and make "
                  "sure %s is wrapped by `paddle::Optional`",
                  in_name,
                  in_name));
          VLOG(3) << "Custom Operator: GradOpDescMaker - handle unfound input: "
                  << in_name;
        }
      }
    }
    for (auto& out_name : outputs_) {
      // Handle inplace optional case
      if (!this->HasInput(detail::NoGrad(out_name, is_double_grad_))) {
        PADDLE_ENFORCE(
            out_name.find(paddle::kOptionalSuffix) != std::string::npos,
            phi::errors::InvalidArgument(
                "Custom operator couldn't find grad operator output name for "
                "%s. If you are using inplace optional inputs & outputs, "
                "please check your InplaceMap and `Outputs` again and make "
                "sure %s is wrapped by `paddle::Optional`",
                out_name,
                out_name));
        VLOG(3) << "Custom Operator: GradOpDescMaker - handle unfound output: "
                << out_name;
        continue;
      }
      VLOG(3) << "Custom Operator: GradOpDescMaker - output: " << out_name;
      if (detail::IsDuplicableVar(out_name)) {
        grad_op->SetOutput(
            out_name,
            this->InputGrad(detail::NoGrad(out_name, is_double_grad_),
                            /*drop_empty_grad=*/false));
      } else {
        grad_op->SetOutput(
            out_name,
            this->InputGrad(detail::NoGrad(out_name, is_double_grad_)));
      }
    }
    grad_op->SetAttrMap(this->Attrs());
  }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  bool is_double_grad_{false};
};

template <>
class CustomGradOpMaker<imperative::OpBase>
    : public SingleGradOpMaker<imperative::OpBase> {
 public:
  explicit CustomGradOpMaker(
      const std::string& type,
      const imperative::NameVarBaseMap& var_base_map_in,
      const imperative::NameVarBaseMap& var_base_map_out,
      const AttributeMap& attrs,
      const std::map<std::string, std::string>& inplace_map,
      const std::string& name,
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs,
      bool is_double_grad)
      : SingleGradOpMaker<imperative::OpBase>(
            type, var_base_map_in, var_base_map_out, attrs, inplace_map),
        name_(name),
        inputs_(inputs),
        outputs_(outputs),
        is_double_grad_(is_double_grad) {}

 protected:
  // TODO(chenweihang): The code is duplicated with the previous one, because
  // ere OpMaker's Input, Output and other methods are protected. Putting the
  // function implementation outside the class will cause the method to be
  // uncallable,
  // so it is still implemented in the class for the time being.
  void Apply(GradOpPtr<imperative::OpBase> grad_op) const override {
    grad_op->SetType(name_);

    auto fwd_op_inputs = this->InputNames();
    auto fwd_op_outputs = this->OutputNames();

    for (auto& in_name : inputs_) {
      VLOG(3) << "Custom Operator: GradOpBaseMaker - input: " << in_name;
      if (!detail::IsGradVar(in_name, is_double_grad_)) {
        if (detail::IsMemberOf(fwd_op_inputs, in_name)) {
          grad_op->SetInput(in_name, this->Input(in_name));
        } else if (detail::IsMemberOf(fwd_op_outputs, in_name)) {
          grad_op->SetInput(in_name, this->Output(in_name));
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "The input tensor name `%s` is invalid, expected it is the input "
              "or output of forward operator.",
              in_name));
        }
      } else {
        // Handle inplace optional case
        if (this->HasOutput(detail::NoGrad(in_name))) {
          grad_op->SetInput(in_name, this->OutputGrad(detail::NoGrad(in_name)));
        } else {
          PADDLE_ENFORCE(
              in_name.find(paddle::kOptionalSuffix) != std::string::npos,
              phi::errors::InvalidArgument(
                  "Custom operator couldn't find grad operator input name for "
                  "%s. If you are using inplace optional inputs & outputs, "
                  "please check your InplaceMap and `Outputs` again and make "
                  "sure %s is wrapped by `paddle::Optional`",
                  in_name,
                  in_name));
          VLOG(3) << "Custom Operator: GradOpBaseMaker - handle unfound input: "
                  << in_name;
        }
      }
    }
    for (auto& out_name : outputs_) {
      // Handle inplace optional case
      if (!this->HasInput(detail::NoGrad(out_name, is_double_grad_))) {
        PADDLE_ENFORCE(
            out_name.find(paddle::kOptionalSuffix) != std::string::npos,
            phi::errors::InvalidArgument(
                "Custom operator couldn't find grad operator output name for "
                "%s. If you are using inplace optional inputs & outputs, "
                "please check your InplaceMap and `Outputs` again and make "
                "sure %s is wrapped by `paddle::Optional`",
                out_name,
                out_name));
        VLOG(3) << "Custom Operator: GradOpBaseMaker - handle unfound output: "
                << out_name;
        continue;
      }
      VLOG(3) << "Custom Operator: GradOpBaseMaker - output: " << out_name;
      grad_op->SetOutput(
          out_name, this->InputGrad(detail::NoGrad(out_name, is_double_grad_)));
    }
    grad_op->SetAttrMap(this->Attrs());
  }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  bool is_double_grad_{false};
};

//////////// Operator and Kernel Register //////////////

static void RegisterOperatorKernelWithPlace(
    const std::string& name,
    const OperatorWithKernel::OpKernelFunc& op_kernel_func,
    const proto::VarType::Type type,
    const platform::Place& place) {
  OpKernelType key(type, place);
  VLOG(3) << "Custom Operator: op kernel key: " << key;
  OperatorWithKernel::AllOpKernels()[name][key] = op_kernel_func;
}

static void RegisterOperatorKernel(
    const std::string& name,
    const paddle::KernelFunc& kernel_func,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& attrs,
    const std::unordered_map<std::string, std::string>& inplace_map,
    void* dso_handle) {
  VLOG(3) << "Custom Operator: op name in kernel: " << name;
  // NOTE [ Dummy Op Kernel Key ]
  // TODO(chenweihang): Because execute engine need get device context based
  // op_kernel_key.place_, so we should register kernel for each
  // device. But this is not entirely correct, if user only give a cpu kernel,
  // but call api in gpu device, it will cause error.
  OperatorWithKernel::OpKernelFunc op_kernel_func;
  if (kernel_func) {
    VLOG(3) << "Register custom operator " << name << " with kernel func";
    op_kernel_func = [kernel_func, inputs, outputs, attrs, inplace_map](
                         const framework::ExecutionContext& ctx) {
      VLOG(3) << "Custom Operator: run custom kernel func in lambda.";
      RunKernelFunc(ctx, kernel_func, inputs, outputs, attrs, inplace_map);
    };
  } else {
    VLOG(3) << "Register custom operator " << name
            << " with raw op kernel func";
    PADDLE_ENFORCE_NOT_NULL(
        dso_handle,
        platform::errors::InvalidArgument(
            "The dso handle must be provided if kernel_func is nullptr."));
    using OpKernelFuncPtr = void(const framework::ExecutionContext&);
    auto symbol_name = "PD_" + name + "_raw_op_kernel_func";
    auto* func = detail::DynLoad<OpKernelFuncPtr>(dso_handle, symbol_name);
    op_kernel_func = func;
  }
  RegisterOperatorKernelWithPlace(
      name, op_kernel_func, proto::VarType::RAW, platform::CPUPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  RegisterOperatorKernelWithPlace(
      name, op_kernel_func, proto::VarType::RAW, platform::CUDAPlace());
#endif
#if defined(PADDLE_WITH_XPU)
  RegisterOperatorKernelWithPlace(
      name, op_kernel_func, proto::VarType::RAW, platform::XPUPlace());
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  auto device_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  for (const auto& dev_type : device_types) {
    for (size_t dev_id = 0;
         dev_id < phi::DeviceManager::GetDeviceCount(dev_type);
         dev_id++) {
      RegisterOperatorKernelWithPlace(name,
                                      op_kernel_func,
                                      proto::VarType::RAW,
                                      platform::CustomPlace(dev_type, dev_id));
    }
  }
#endif
}

void RegisterOperatorWithMetaInfo(const std::vector<OpMetaInfo>& op_meta_infos,
                                  void* dso_handle) {
  /* Op register */
  OpInfo info;

  auto& base_op_meta = op_meta_infos.front();

  auto op_name = OpMetaInfoHelper::GetOpName(base_op_meta);

  if (OpInfoMap::Instance().Has(op_name)) {
    LOG(WARNING) << "Operator (" << op_name << ") has been registered.";
    return;
  }

  auto& op_inputs = OpMetaInfoHelper::GetInputs(base_op_meta);
  auto& op_outputs = OpMetaInfoHelper::GetOutputs(base_op_meta);
  auto& op_attrs = OpMetaInfoHelper::GetAttrs(base_op_meta);
  auto& op_inplace_map = OpMetaInfoHelper::GetInplaceMap(base_op_meta);
  auto& op_inplace_reverse_map =
      OpMetaInfoHelper::GetInplaceReverseMap(base_op_meta);
  auto& kernel_fn = OpMetaInfoHelper::GetKernelFn(base_op_meta);
  auto& infer_shape_func = OpMetaInfoHelper::GetInferShapeFn(base_op_meta);
  auto& infer_dtype_func = OpMetaInfoHelper::GetInferDtypeFn(base_op_meta);

  VLOG(3) << "Custom Operator: forward, op name: " << op_name;
  VLOG(3) << "Custom Operator: forward, op inputs: "
          << string::join_strings(op_inputs, ',');
  VLOG(3) << "Custom Operator: forward, op outputs: "
          << string::join_strings(op_outputs, ',');
  VLOG(3) << "Custom Operator: forward, op attrs: "
          << string::join_strings(op_attrs, ',');
  if (!op_inplace_map.empty()) {
    VLOG(3) << "Custom Operator: forward, op inplace_map: "
            << string::join_strings(op_inplace_map, ',', [](auto& pair) {
                 return pair.first + ": " + pair.second;
               });
  }

  // Op
  info.creator_ = [](const std::string& op_name,
                     const VariableNameMap& inputs,
                     const VariableNameMap& outputs,
                     const AttributeMap& attrs) {
    return new CustomOperator(op_name, inputs, outputs, attrs);
  };

  // OpMaker
  info.proto_ = new proto::OpProto;
  info.proto_->set_type(op_name);

  info.checker_ = new OpAttrChecker();
  CustomOpMaker custom_maker(op_inputs, op_outputs, op_attrs);
  custom_maker(info.proto_, info.checker_);
  PADDLE_ENFORCE_EQ(
      info.proto_->IsInitialized(),
      true,
      platform::errors::PreconditionNotMet(
          "Fail to initialize %s's OpProto, because %s is not initialized.",
          op_name,
          info.proto_->InitializationErrorString()));

  // Inplace
  if (!op_inplace_map.empty()) {
    info.infer_inplace_ = [op_inplace_map](bool use_cuda) {
      return op_inplace_map;
    };
  }

  // InferShape
  if (infer_shape_func == nullptr) {
    // use default InferShape
    info.infer_shape_ =
        [op_inputs, op_outputs, op_inplace_map](InferShapeContext* ctx) {
          RunDefaultInferShapeFunc(ctx, op_inputs, op_outputs, op_inplace_map);
        };
  } else {
    info.infer_shape_ = [op_inputs,
                         op_outputs,
                         op_attrs,
                         op_inplace_map,
                         op_inplace_reverse_map,
                         infer_shape_func](InferShapeContext* ctx) {
      RunInferShapeFunc(ctx,
                        infer_shape_func,
                        op_inputs,
                        op_outputs,
                        op_attrs,
                        op_inplace_map,
                        op_inplace_reverse_map);
    };
  }

  // Infer Dtype
  if (infer_dtype_func == nullptr) {
    // use default InferDtype
    info.infer_var_type_ =
        [op_inputs, op_outputs, op_inplace_map](InferVarTypeContext* ctx) {
          RunDefaultInferDtypeFunc(ctx, op_inputs, op_outputs, op_inplace_map);
        };
  } else {
    info.infer_var_type_ = [op_inputs,
                            op_outputs,
                            op_inplace_map,
                            op_inplace_reverse_map,
                            infer_dtype_func](InferVarTypeContext* ctx) {
      RunInferDtypeFunc(ctx,
                        infer_dtype_func,
                        op_inputs,
                        op_outputs,
                        op_inplace_map,
                        op_inplace_reverse_map);
    };
  }

  // Kernel func
  RegisterOperatorKernel(op_name,
                         kernel_fn,
                         op_inputs,
                         op_outputs,
                         op_attrs,
                         op_inplace_map,
                         dso_handle);

  // If grad op or double grad op exists
  std::string cur_op_name = op_name;
  for (size_t i = 1; i < op_meta_infos.size(); ++i) {
    auto& cur_grad_op = op_meta_infos[i];

    auto& grad_op_name = OpMetaInfoHelper::GetOpName(cur_grad_op);
    auto& grad_op_inputs = OpMetaInfoHelper::GetInputs(cur_grad_op);
    auto& grad_op_outputs = OpMetaInfoHelper::GetOutputs(cur_grad_op);
    auto& grad_op_attrs = OpMetaInfoHelper::GetAttrs(cur_grad_op);
    auto& grad_op_inplace_map = OpMetaInfoHelper::GetInplaceMap(cur_grad_op);
    auto& grad_op_inplace_reverse_map =
        OpMetaInfoHelper::GetInplaceReverseMap(cur_grad_op);
    auto& grad_kernel_fn = OpMetaInfoHelper::GetKernelFn(cur_grad_op);
    auto& grad_infer_shape_fn = OpMetaInfoHelper::GetInferShapeFn(cur_grad_op);

    VLOG(3) << "Custom Operator: backward, op name: " << grad_op_name;
    VLOG(3) << "Custom Operator: backward, op inputs: "
            << string::join_strings(grad_op_inputs, ',');
    VLOG(3) << "Custom Operator: backward, op outputs: "
            << string::join_strings(grad_op_outputs, ',');
    VLOG(3) << "Custom Operator: backward, op attrs: "
            << string::join_strings(grad_op_attrs, ',');
    if (!op_inplace_map.empty()) {
      VLOG(3) << "Custom Operator: backward, op inplace_map: "
              << string::join_strings(grad_op_inplace_map, ',', [](auto& pair) {
                   return pair.first + ": " + pair.second;
                 });
    }

    bool is_double_grad = (i == 2);

    // GradOpDescMaker
    info.grad_op_maker_ =
        [grad_op_name, grad_op_inputs, grad_op_outputs, is_double_grad](
            const OpDesc& fwd_op,
            const std::unordered_set<std::string>& no_grad_set,
            std::unordered_map<std::string, std::string>* grad_to_var,
            const std::vector<BlockDesc*>& grad_block) {
          CustomGradOpMaker<paddle::framework::OpDesc> maker(fwd_op,
                                                             no_grad_set,
                                                             grad_to_var,
                                                             grad_block,
                                                             grad_op_name,
                                                             grad_op_inputs,
                                                             grad_op_outputs,
                                                             is_double_grad);
          return maker();
        };

    // GradOpBaseMaker
    info.dygraph_grad_op_maker_ =
        [grad_op_name, grad_op_inputs, grad_op_outputs, is_double_grad](
            const std::string& type,
            const imperative::NameVarBaseMap& var_base_map_in,
            const imperative::NameVarBaseMap& var_base_map_out,
            const framework::AttributeMap& attrs,
            const framework::AttributeMap& default_attrs,
            const std::map<std::string, std::string>& inplace_map) {
          CustomGradOpMaker<paddle::imperative::OpBase> maker(type,
                                                              var_base_map_in,
                                                              var_base_map_out,
                                                              attrs,
                                                              inplace_map,
                                                              grad_op_name,
                                                              grad_op_inputs,
                                                              grad_op_outputs,
                                                              is_double_grad);
          maker.SetDygraphDefaultAttrsMap(default_attrs);
          return maker();
        };

    /* Grad op register */
    OpInfo grad_info;

    // Grad Op
    grad_info.creator_ = [](const std::string& type,
                            const VariableNameMap& inputs,
                            const VariableNameMap& outputs,
                            const AttributeMap& attrs) {
      return new CustomOperator(type, inputs, outputs, attrs);
    };

    // Inplace
    if (!grad_op_inplace_map.empty()) {
      grad_info.infer_inplace_ = [grad_op_inplace_map](bool use_cuda) {
        return grad_op_inplace_map;
      };
    }

    // Grad InferShape
    if (grad_infer_shape_fn == nullptr) {
      grad_info.infer_shape_ = [grad_op_inputs,
                                grad_op_outputs,
                                is_double_grad](InferShapeContext* ctx) {
        // 1. if forward input exists, gradient's shape is same with forward
        // input
        // default
        //    [Suitable for most situations]
        // 2. if forward input not exists, and only contains one grad input and
        // output,
        //    use grad input shape as grad output shape
        //    [Suitable for the situation that forward input is not used as
        //    backward input]
        for (auto& out_name : grad_op_outputs) {
          auto fwd_name = detail::NoGrad(out_name, is_double_grad);
          if (detail::IsDuplicableVar(fwd_name)) {
            // Duplicable forward var must as backward input
            ctx->ShareDim(fwd_name, out_name);
          } else {
            if (ctx->HasInput(fwd_name)) {
              ctx->ShareDim(fwd_name, out_name);
            } else {
              PADDLE_ENFORCE_EQ(
                  grad_op_inputs.size() == 1UL && grad_op_outputs.size() == 1UL,
                  true,
                  platform::errors::Unavailable(
                      "Custom grad operator infershape error. "
                      "If a custom grad operator contains only one input and "
                      "only one output, the input shape will be directly set "
                      "to the output shape. Otherwise, Please set the forward "
                      "input as the grad operator's input or set the "
                      "InferShapeFn of custom grad operator by "
                      ".SetInferShapeFn(PD_INFER_SHAPE(...))"));
              ctx->ShareDim(grad_op_inputs[0], out_name);
            }
          }
        }
      };
    } else {
      grad_info.infer_shape_ = [grad_op_inputs,
                                grad_op_outputs,
                                grad_op_attrs,
                                grad_op_inplace_map,
                                grad_op_inplace_reverse_map,
                                grad_infer_shape_fn](InferShapeContext* ctx) {
        RunInferShapeFunc(ctx,
                          grad_infer_shape_fn,
                          grad_op_inputs,
                          grad_op_outputs,
                          grad_op_attrs,
                          grad_op_inplace_map,
                          grad_op_inplace_reverse_map);
      };
    }

    // Kernel func
    RegisterOperatorKernel(grad_op_name,
                           grad_kernel_fn,
                           grad_op_inputs,
                           grad_op_outputs,
                           grad_op_attrs,
                           grad_op_inplace_map,
                           dso_handle);

    // update current info
    OpInfoMap::Instance().Insert(cur_op_name, info);
    cur_op_name = grad_op_name;
    info = grad_info;
  }
  // insert last info
  OpInfoMap::Instance().Insert(cur_op_name, info);
}

void RegisterOperatorWithMetaInfoMap(
    const paddle::OpMetaInfoMap& op_meta_info_map, void* dso_handle) {
  auto& meta_info_map = op_meta_info_map.GetMap();
  VLOG(3) << "Custom Operator: size of op meta info map - "
          << meta_info_map.size();
  // pair: {op_type, OpMetaInfo}
  for (auto& pair : meta_info_map) {
    VLOG(3) << "Custom Operator: pair first -> op name: " << pair.first;
    RegisterOperatorWithMetaInfo(pair.second, dso_handle);
  }
}

////////////////////// User APIs ///////////////////////

// load op api
const std::unordered_map<std::string, std::vector<OpMetaInfo>>&
LoadOpMetaInfoAndRegisterOp(const std::string& dso_name) {
  void* handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);
  VLOG(3) << "load custom_op lib: " << dso_name;
  typedef OpMetaInfoMap& get_op_meta_info_map_t();
  auto* get_op_meta_info_map =
      detail::DynLoad<get_op_meta_info_map_t>(handle, "PD_GetOpMetaInfoMap");
  auto& op_meta_info_map = get_op_meta_info_map();
  RegisterOperatorWithMetaInfoMap(op_meta_info_map, handle);
  return op_meta_info_map.GetMap();
}

}  // namespace framework
}  // namespace paddle
