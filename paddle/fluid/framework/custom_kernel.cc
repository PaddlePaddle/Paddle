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

#ifdef PADDLE_WITH_CUSTOM_KERNEL

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

namespace framework {

namespace detail {

// dynamic lib open
static inline void* DynOpen(const std::string& dso_name) {
  void* dso_handle = nullptr;
  int dynload_flags = RTLD_LAZY | RTLD_LOCAL;
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
  auto errorno = dlerror();
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
static void RunKernelFunc(pten::KernelContext* ctx,
                          const OpKernelInfo& op_kernel_info) {
  VLOG(3) << "[CUSTOM KERNEL] RunKernelFunc begin...";

  // input and output size is not params' num
  // but actual Tensors' size
  size_t input_size = ctx->InputsSize();
  size_t output_size = ctx->OutputsSize();
  size_t attr_size = ctx->AttrsSize();

  // parameters' num of unified user kernel function
  auto& input_defs = op_kernel_info.input_defs();
  auto& output_defs = op_kernel_info.output_defs();
  auto& attribute_defs = op_kernel_info.attribute_defs();

  PADDLE_ENFORCE_GE(input_size, input_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of ctx inputs size (%d) must be larger than "
                        "the size of kernel input_defs (%d).",
                        input_size, input_defs.size()));

  PADDLE_ENFORCE_GE(output_size, output_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of ctx outputs size (%d) must be larger than "
                        "the size of kernel output_defs (%d).",
                        output_size, output_defs.size()));

  PADDLE_ENFORCE_EQ(attr_size, attribute_defs.size(),
                    platform::errors::InvalidArgument(
                        "the size of ctx attribute size (%d) must be equal to "
                        "to the size of kernel attribute_defs (%d).",
                        attr_size, attribute_defs.size()));

  VLOG(3) << "[CUSTOM KERNEL] InputSize: " << input_defs.size()
          << " AttrsSize: " << attribute_defs.size()
          << " OutputSize: " << output_defs.size();

  // Inputs mapping
  std::vector<paddle::experimental::Tensor> custom_ins;
  std::vector<std::vector<paddle::experimental::Tensor>> custom_vec_ins;
  for (size_t in_idx = 0; in_idx < input_defs.size(); ++in_idx) {
    VLOG(3) << "Mapping Input[" << in_idx << "]";
    const std::pair<int, int> range = ctx->InputRangeAt(in_idx);
    // is_vector tells if this Input is Tensor or std::vector<Tensor>
    if (!input_defs.at(in_idx).is_vector) {
      paddle::experimental::Tensor custom_t(ctx->InputSharedPtrAt(range.first));
      custom_ins.push_back(custom_t);
      VLOG(3) << "Mapped Tensor Input[" << in_idx
              << "] with range.first: " << range.first;
    } else {
      std::vector<paddle::experimental::Tensor> custom_vec_in;
      auto ctx_tensor_vec =
          ctx->InputSharedPtrBetween(range.first, range.second);
      for (auto& ctx_tensor : ctx_tensor_vec) {
        paddle::experimental::Tensor custom_t(ctx_tensor);
        custom_vec_in.push_back(custom_t);
      }
      custom_vec_ins.push_back(custom_vec_in);
      VLOG(3) << "Mapped std::vector<Tensor> Input[" << in_idx
              << "] with range.first: " << range.first
              << ", and rang.second: " << range.second;
    }
  }

  // Attributes mapping
  std::vector<paddle::any> custom_attrs;
  for (size_t attr_idx = 0; attr_idx < attribute_defs.size(); ++attr_idx) {
    VLOG(3) << "Mapping Attribute[" << attr_idx << "]";
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
    VLOG(3) << "Mapped Attribute[" << attr_idx << "]";
  }

  // Outputs mapping
  std::vector<paddle::experimental::Tensor*> custom_outs;
  std::vector<std::vector<paddle::experimental::Tensor*>> custom_vec_outs;
  for (size_t out_idx = 0; out_idx < output_defs.size(); ++out_idx) {
    VLOG(3) << "Mapping Output[" << out_idx << "]";
    const std::pair<int, int> range = ctx->OutputRangeAt(out_idx);
    // is_vector tells if this Output is Tensor or std::vector<Tensor>
    if (!output_defs.at(out_idx).is_vector) {
      auto ctx_tensor = ctx->OutputSharedPtrAt(range.first);
      paddle::experimental::Tensor* custom_t =
          new paddle::experimental::Tensor(ctx_tensor);
      custom_outs.push_back(custom_t);
      VLOG(3) << "Mapped Tensor Output[" << out_idx
              << "] with range.first: " << range.first;
    } else {
      std::vector<paddle::experimental::Tensor*> custom_vec_out;
      auto ctx_tensor_vec =
          ctx->OutputSharedPtrBetween(range.first, range.second);
      for (auto& ctx_tensor : ctx_tensor_vec) {
        paddle::experimental::Tensor* custom_t =
            new paddle::experimental::Tensor(ctx_tensor);
        custom_vec_out.push_back(custom_t);
      }
      custom_vec_outs.push_back(custom_vec_out);
      VLOG(3) << "Mapped std::vector<Tensor> Output[" << out_idx
              << "] with range.first: " << range.first
              << ", and rang.second: " << range.second;
    }
  }

  // DeviceContext:
  // In pten, kernel function knows XXContext through template param
  // from input KernelContext, but here we don't have KernelContext
  // and need mapping to user kernel function with backend from OpKernelInfo
  // user_kernel_fn will do static_cast according to user kernel function.
  // we just set necessary info to dev_ctx(such as stream in NPUContext) before
  // pten::DeviceContext is exposed to outer
  DeviceContext dev_ctx;
  if (op_kernel_info.GetBackend() == pten::Backend::CPU) {
    // do nothing
  } else {
    LOG(ERROR) << "[CUSTOM KERNEL] mismatched kernel backend: "
               << op_kernel_info.GetBackend() << " with compiled paddle.";
    return;
  }
  auto& user_kernel_fn = OpKernelInfoHelper::GetKernelFn(op_kernel_info);
  // call user function
  user_kernel_fn(dev_ctx, custom_ins, custom_vec_ins, custom_attrs,
                 &custom_outs, &custom_vec_outs);
  VLOG(3) << "[CUSTOM KERNEL] finished call user kernel function";

  // delete newed paddle::Tensor for calling user kernel function
  for (size_t i = 0; i < custom_outs.size(); ++i) {
    delete custom_outs[i];
  }
  for (size_t i = 0; i < custom_vec_outs.size(); ++i) {
    for (size_t j = 0; j < custom_vec_outs[i].size(); ++j) {
      delete custom_vec_outs[i][j];
    }
  }
}

void RegisterKernelWithMetaInfo(
    const std::vector<OpKernelInfo>& op_kernel_infos) {
  PADDLE_ENFORCE_EQ(FLAGS_run_pten_kernel, true,
                    platform::errors::Unimplemented(
                        "Custom Kernel depends on pten kernel enabled,"));

  for (size_t i = 0; i < op_kernel_infos.size(); ++i) {
    auto& kernel_info = op_kernel_infos[i];
    auto op_type = OpKernelInfoHelper::GetOpName(kernel_info);
    auto kernel_key = OpKernelInfoHelper::GetKernelKey(kernel_info);

    VLOG(3) << "[CUSTOM KERNEL] registering [" << op_type << "]" << kernel_key;

    // 1.Check whether this kernel is valid for a specific operator
    PADDLE_ENFORCE_EQ(
        pten::KernelFactory::Instance().HasCompatiblePtenKernel(op_type), true,
        platform::errors::InvalidArgument(
            "[CUSTOM KERNEL] %s is not ready for custom kernel registering",
            op_type));

    // 2.Check whether kernel_key has been already registed
    if (pten::KernelFactory::Instance().kernels()[op_type].find(kernel_key) !=
        pten::KernelFactory::Instance().kernels()[op_type].end()) {
      LOG(WARNING)
          << "[CUSTOM KERNEL] The operator <" << op_type << ">'s kernel "
          << kernel_key << " has been already existed in Paddle, "
          << "please contribute PR if need to optimize the kernel code. "
          << "Custom kernel do NOT support to replace existing kernel in "
             "Paddle.";
      continue;
    }

    // pten::KernelFn
    pten::KernelFn kernel_fn = [kernel_info](pten::KernelContext* ctx) {
      VLOG(3) << "[CUSTOM KERNEL] run custom PTEN kernel func in lambda.";
      RunKernelFunc(ctx, kernel_info);
    };
    // variadic_kernel_fn
    void* variadic_kernel_fn =
        OpKernelInfoHelper::GetVariadicKernelFn(kernel_info);
    pten::Kernel kernel(kernel_fn, variadic_kernel_fn);
    // args info
    ParseArgs(kernel_info, kernel.mutable_args_def());
    // register custom kernel to pten::KernelFactory
    pten::KernelFactory::Instance().kernels()[op_type][kernel_key] = kernel;
    VLOG(3) << "[CUSTOM KERNEL] Successed in registering operator <" << op_type
            << ">'s kernel " << kernel_key << " to Paddle. "
            << "It will be used like native ones";
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

// Try loading custom kernel from PADDLE_PLUGIN_ROOT
// PADDLE_CUSTOM_KERNEL is an abstract path of custom kernel library, such
// as '/path/to/libmy_custom_kernel.so', default value is ''
// TODO(SJC): common loading with pluggable device
void TryLoadCustomKernel() {
  const char* env_dso_ptr = std::getenv("PADDLE_PLUGIN_ROOT");
  if (env_dso_ptr == nullptr) {
    VLOG(3) << "[CUSTOM KERNEL] PADDLE_PLUGIN_ROOT is not set";
    return;
  }
  std::string dso_name(env_dso_ptr);
  if (dso_name.empty()) {
    VLOG(3) << "[CUSTOM KERNEL] PADDLE_PLUGIN_ROOT is empty";
    return;
  }
  VLOG(3) << "[CUSTOM KERNEL] PADDLE_PLUGIN_ROOT=" << dso_name;

  LoadOpKernelInfoAndRegister(dso_name);
}

}  // namespace framework
}  // namespace paddle

#endif  // PADDLE_WITH_CUSTOM_KERNEL
