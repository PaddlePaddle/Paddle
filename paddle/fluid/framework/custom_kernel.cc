/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/custom_kernel.h"
#include "paddle/fluid/framework/op_kernel_info_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/pten/api/ext/op_kernel_info.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/kernel_registry.h"

DECLARE_bool(run_pten_kernel);

namespace paddle {

namespace platform {
#ifdef PADDLE_WITH_ASCEND_CL
class NPUDeviceContext;
#endif
}  // namespace platform

namespace framework {

#ifdef PADDLE_WITH_ASCEND_CL
using NPUContext = paddle::platform::NPUDeviceContext;
#endif

namespace detail {

// dynamic lib open
static inline void* DynOpen(const std::string& dso_name) {
  void* dso_handle = nullptr;
#if !defined(_WIN32)
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
#else
  int dynload_flags = 0;
#endif  // !_WIN32
  dso_handle = dlopen(dso_name.c_str(), dynload_flags);
  PADDLE_ENFORCE_NOT_NULL(
      dso_handle,
      platform::errors::InvalidArgument("Fail to open library: %s", dso_name));
  return dso_handle;
}

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
      func, platform::errors::NotFound(
                "Failed to load dynamic operator library, error message(%s).",
                errorno));
  return func;
}

}  // namespace detail

// set pten::Kernel args_def_ from op_kernel_info
// because we can not set directly to pten::Kernel without exposing
// pten::KernelArgsDef
static void ParseArgs(const OpKernelInfo& op_kernel_info,
                      pten::KernelArgsDef* args_def) {
  auto& input_defs = op_kernel_info.input_defs();
  auto& output_defs = op_kernel_info.output_defs();
  auto& attribute_defs = op_kernel_info.attribute_defs();

  for (auto& input : input_defs) {
    args_def->AppendInput(input.backend, input.layout, input.dtype);
  }
  for (auto& output : output_defs) {
    args_def->AppendOutput(output.backend, output.layout, output.dtype);
  }
  for (auto& attr : attribute_defs) {
    args_def->AppendAttribute(attr.type_index);
  }
}

// custom pten kernel call function define
static void RunKernelFunc(const OpKernelInfo& op_kernel_info,
                          pten::KernelContext* ctx,
                          const CustomKernelFunc& func) {
  VLOG(3) << "[CUSTOM KERNEL] RunKernelFunc";

  size_t input_size = ctx->InputsSize();
  size_t output_size = ctx->OutputsSize();
  size_t attr_size = ctx->AttrsSize();

  // input:
  std::vector<paddle::experimental::Tensor> custom_ins;
  std::vector<std::vector<paddle::experimental::Tensor>> custom_vec_ins;
  for (size_t in_idx = 0; in_idx < input_size; ++in_idx) {
    const std::pair<int, int> range = ctx->InputRangeAt(in_idx);
    if (range.first + 1 == range.second) {
      paddle::experimental::Tensor custom_t;
      auto ctx_tensor = ctx->InputAt<pten::DenseTensor>(range.first);
      custom_t.set_impl(std::make_unique<pten::DenseTensor>(ctx_tensor));
      custom_ins.push_back(custom_t);
    } else {
      std::vector<paddle::experimental::Tensor> custom_vec_in;
      auto ctx_tensor_vec =
          ctx->MoveInputsBetween<pten::DenseTensor>(range.first, range.second);
      for (auto& ctx_tensor : ctx_tensor_vec) {
        paddle::experimental::Tensor custom_t;
        custom_t.set_impl(std::make_unique<pten::DenseTensor>(ctx_tensor));
        custom_vec_in.push_back(custom_t);
      }
      custom_vec_ins.push_back(custom_vec_in);
    }
  }
  // attr:
  std::vector<paddle::any> custom_attrs;
  auto attribute_defs = op_kernel_info.attribute_defs();
  for (size_t attr_idx = 0; attr_idx < attr_size; ++attr_idx) {
    if (attribute_defs[attr_idx].type_index == std::type_index(typeid(bool))) {
      bool arg = ctx->AttrAt<bool>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(int))) {
      int arg = ctx->AttrAt<int>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(float))) {
      float arg = ctx->AttrAt<float>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(double))) {
      double arg = ctx->AttrAt<double>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(int64_t))) {
      int64_t arg = ctx->AttrAt<int64_t>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(paddle::platform::float16))) {
      paddle::platform::float16 arg =
          ctx->AttrAt<paddle::platform::float16>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(DataType))) {
      DataType arg = ctx->AttrAt<DataType>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(Scalar))) {
      Scalar arg = ctx->AttrAt<Scalar>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(std::vector<int64_t>))) {
      std::vector<int64_t> arg = ctx->AttrAt<std::vector<int64_t>>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(ScalarArray))) {
      ScalarArray arg = ctx->AttrAt<ScalarArray>(attr_idx);
      custom_attrs.push_back(arg);
    } else if (attribute_defs[attr_idx].type_index ==
               std::type_index(typeid(std::vector<int>))) {
      std::vector<int> arg = ctx->AttrAt<std::vector<int>>(attr_idx);
      custom_attrs.push_back(arg);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported attribute attribute_defs[%d].type_index", attr_idx));
    }
  }
  // output:
  std::vector<paddle::experimental::Tensor*> custom_outs;
  std::vector<std::vector<paddle::experimental::Tensor*>> custom_vec_outs;
  for (size_t out_idx = 0; out_idx < output_size; ++out_idx) {
    const std::pair<int, int> range = ctx->OutputRangeAt(out_idx);
    if (range.first + 1 == range.second) {
      auto ctx_tensor = ctx->MutableSharedOutputAt(range.first);
      paddle::experimental::Tensor custom_t;
      custom_t.set_impl(ctx_tensor);
      custom_outs.push_back(&custom_t);
    } else {
      std::vector<paddle::experimental::Tensor*> custom_vec_out;
      auto ctx_tensor_vec =
          ctx->MutableSharedOutputBetween(range.first, range.second);
      for (auto& ctx_tensor : ctx_tensor_vec) {
        paddle::experimental::Tensor custom_t;
        custom_t.set_impl(ctx_tensor);
        custom_vec_out.push_back(&custom_t);
      }
      custom_vec_outs.push_back(custom_vec_out);
    }
  }
  // device_ctx:
  // in pten, kernel function knows XXContext through template param
  // from input KernelContext, but here we don't have KernelContext
  // and need mapping to user kernel function with backend from OpKernelInfo
  DeviceContext dev_ctx;
  if (op_kernel_info.GetBackend() == pten::Backend::CPU) {
// do nothing
#ifdef PADDLE_WITH_ASCEND_CL
  } else if (op_kernel_info.GetBackend() == pten::Backend::NPU) {
    const NPUContext& dev_context = ctx->GetDeviceContext<NPUContext>();
    dev_ctx.set_stream(dev_context.stream());
#endif
  } else {
    LOG(ERROR) << "[CUSTOM KERNEL] mismatched kernel backend: "
               << op_kernel_info.GetBackend() << " with compiled paddle.";
    return;
  }

  // call user function
  func(dev_ctx, custom_ins, custom_vec_ins, custom_attrs, &custom_outs,
       &custom_vec_outs);
  VLOG(3) << "[CUSTOM KERNEL] finished call user kernel function";
}

void RegisterKernelWithMetaInfo(
    const std::vector<OpKernelInfo>& op_kernel_infos) {
  PADDLE_ENFORCE_EQ(FLAGS_run_pten_kernel, true,
                    platform::errors::Unimplemented(
                        "Custom Kernel depend on pten kernel enabled,"));

  for (size_t i = 0; i < op_kernel_infos.size(); ++i) {
    auto& kernel_info = op_kernel_infos[i];
    auto op_type = OpKernelInfoHelper::GetOpName(kernel_info);
    auto kernel_key = OpKernelInfoHelper::GetKernelKey(kernel_info);

    VLOG(3) << "[CUSTOM KERNEL] registering [" << op_type << "]" << kernel_key;

    // 1.Check wether this kernel is valid for a specific operator
    if (!pten::KernelFactory::Instance().HasCompatiblePtenKernel(op_type)) {
      LOG(WARNING) << "[CUSTOM KERNEL] skipped: " << op_type
                   << " is not ready for custom kernel registering.";
      continue;
    }

    // 2.Check wether kernel_key has been already registed
    if (pten::KernelFactory::Instance().kernels()[op_type].find(kernel_key) !=
        pten::KernelFactory::Instance().kernels()[op_type].end()) {
      LOG(WARNING) << "[CUSTOM KERNEL] skipped: " << kernel_key
                   << " has been registered already.";
      continue;
    }

    // pten::KernelFn
    auto& user_kernel_fn = OpKernelInfoHelper::GetKernelFn(kernel_info);
    pten::KernelFn kernel_fn = [kernel_info,
                                user_kernel_fn](pten::KernelContext* ctx) {
      VLOG(3) << "[CUSTOM KERNEL] run custom PTEN kernel func in lambda.";
      RunKernelFunc(kernel_info, ctx, user_kernel_fn);
    };
    // variadic_kernel_fn
    void* variadic_kernel_fn =
        OpKernelInfoHelper::GetVariadicKernelFn(kernel_info);
    pten::Kernel kernel(kernel_fn, variadic_kernel_fn);
    // args info
    ParseArgs(kernel_info, kernel.mutable_args_def());
    // register custom kernel to pten::KernelFactory
    pten::KernelFactory::Instance().kernels()[op_type][kernel_key] = kernel;
    VLOG(3) << "[CUSTOM KERNEL] registered custom PTEN kernel";
  }
}

void RegisterKernelWithMetaInfoMap(
    const paddle::OpKernelInfoMap& op_kernel_info_map) {
  auto& kernel_info_map = op_kernel_info_map.GetMap();
  VLOG(3) << "[CUSTOM KERNEL] size of op_kernel_info_map: "
          << kernel_info_map.size();

  // pair: {op_type, OpKernelInfo}
  for (auto& pair : kernel_info_map) {
    VLOG(3) << "[CUSTOM KERNEL] pair first -> op name: " << pair.first;
    RegisterKernelWithMetaInfo(pair.second);
  }
}

// load custom kernel from dso_name
void LoadOpKernelInfoAndRegister(const std::string& dso_name) {
  VLOG(3) << "[CUSTOM KERNEL] load custom_kernel lib: " << dso_name;

  void* handle = detail::DynOpen(dso_name);

  typedef OpKernelInfoMap& get_op_kernel_info_map_t();
  auto* get_op_kernel_info_map = detail::DynLoad<get_op_kernel_info_map_t>(
      handle, "PD_GetOpKernelInfoMap");

  // If symbol PD_GetOpKernelInfoMap is not in lib, we ignore this lib
  if (get_op_kernel_info_map != nullptr) {
    auto& op_kernel_info_map = get_op_kernel_info_map();
    RegisterKernelWithMetaInfoMap(op_kernel_info_map);
  }
}

// Try loading custom kernel from PADDLE_CUSTOM_KERNEL
// PADDLE_CUSTOM_KERNEL is an abstract path of custom kernel library, such
// as '/path/to/libmy_custom_kernel.so', default value is ''
// TODO(SJC): common loading with pluggable device
void TryLoadCustomKernel() {
  const char* env_dso_ptr = std::getenv("PADDLE_CUSTOM_KERNEL");
  if (env_dso_ptr == nullptr) {
    VLOG(3) << "[CUSTOM KERNEL] PADDLE_CUSTOM_KERNEL is not set";
    return;
  }
  std::string dso_name(env_dso_ptr);
  if (dso_name.empty()) {
    VLOG(3) << "[CUSTOM KERNEL] PADDLE_CUSTOM_KERNEL is empty";
    return;
  }
  VLOG(3) << "[CUSTOM KERNEL] PADDLE_CUSTOM_KERNEL=" << dso_name;

  LoadOpKernelInfoAndRegister(dso_name);
}

}  // namespace framework
}  // namespace paddle
