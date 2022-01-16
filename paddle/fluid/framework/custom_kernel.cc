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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/string/string_helper.h"
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

// custom op kernel call function define
static void RunFluidKernelFunc(const framework::ExecutionContext& ctx,
                               const CustomKernelFunc& func) {
  std::cout << "[CUSTOM KERNEL] Run ComputeFunc." << std::endl;
  /*
    auto input_names = ctx.InNameList();
    auto output_names = ctx.OutNameList();
    auto op_attrs = ctx.Attrs();

    LOG(INFO) << "[CUSTOM KERNEL] inputs name size - " << input_names.size();
    LOG(INFO) << "[CUSTOM KERNEL] outputs name size - " << output_names.size();
    LOG(INFO) << "[CUSTOM KERNEL] attrs name size - " << op_attrs.size();

    // check func inputs / outputs before real execution
    //
    强制用户按照框架内的参数个数和顺序的方式不容易稳定：比如内部kernel增减或修改参数
    // 这也是为何在fluid kernel体系 且
    //
    外部用户的kernel函数为functional时，需要让用户在kernel注册的时候提供参数、属性、返回的原因
    // 外部用户注册时提供信息后可以确保映射关系 但
    // 用户针对kernel也需要指定这些信息太冗杂
    //
    pten方式下用户参照各方都严格遵守的api.yaml或内部同名Kernel的形式实现时，不存在这个问题。

    std::vector<paddle::experimental::Tensor> custom_ins;
    std::vector<std::vector<paddle::experimental::Tensor>> custom_vec_ins;

    for (auto& in_name : input_names) {
      LOG(INFO) << "[CUSTOM KERNEL] input name - " << in_name;
      if (detail::IsDuplicableVar(in_name)) {
        // return const std::vector<const Tensor*>
        auto vec_x = ctx.MultiInput<Tensor>(in_name);
        PADDLE_ENFORCE_NE(vec_x.empty(), true,
                          platform::errors::NotFound(
                              "Input vector<tensor> (%s) is empty.", in_name));
        std::vector<paddle::experimental::Tensor> custom_vec_in;
        for (size_t i = 0; i < vec_x.size(); ++i) {
          auto* x = vec_x[i];
          PADDLE_ENFORCE_NOT_NULL(
              x, platform::errors::NotFound(
                     "The %d-th tensor in input vector<tensor> (%s) is
    nullptr.",
                     i, in_name));
          PADDLE_ENFORCE_EQ(x->IsInitialized(), true,
                            platform::errors::InvalidArgument(
                                "The %d-th tensor in input vector<tensor> (%s) "
                                "is not initialized.",
                                i, in_name));
          paddle::experimental::Tensor custom_t;
          custom_t.set_impl(std::move(experimental::MakePtenDenseTensor(*x)));
          custom_vec_in.emplace_back(custom_t);
        }
        custom_vec_ins.emplace_back(custom_vec_in);
      } else {
        auto* x = ctx.Input<Tensor>(in_name);
        PADDLE_ENFORCE_NOT_NULL(x, platform::errors::NotFound(
                                       "Input tensor (%s) is nullptr.",
    in_name));
        PADDLE_ENFORCE_EQ(x->IsInitialized(), true,
                          platform::errors::InvalidArgument(
                              "Input tensor (%s) is not initialized.",
    in_name));
        paddle::experimental::Tensor custom_in;
        custom_in.set_impl(std::move(experimental::MakePtenDenseTensor(*x)));
        custom_ins.emplace_back(custom_in);
      }
    }

    std::vector<paddle::any> custom_attrs;
    for (auto& op_attr : op_attrs) {
      auto attr_name = op_attr.first;
      LOG(INFO) << "[CUSTOM KERNEL] attr name - " << attr_name;
      auto attr = op_attr.second;
      if (attr.type() == typeid(bool)) {
        custom_attrs.emplace_back(ctx.Attr<bool>(attr_name));
      } else if (attr.type() == typeid(int)) {
        custom_attrs.emplace_back(ctx.Attr<int>(attr_name));
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported `%s` type value as custom attribute now. "
            "Supported data types include `bool`, `int`, `float`, "
            "`int64_t`, `std::string`, `std::vector<int>`, "
            "`std::vector<float>`, `std::vector<int64_t>`, "
            "`std::vector<std::string>`, Please check whether "
            "the attribute data type and data type string are matched.",
            attr_name));
      }
    }

    LOG(INFO) << "[CUSTOM KERNEL] Run ComputeFunc.";
    try {
      auto outs = func(custom_ins, custom_vec_ins, custom_attrs);

      LOG(INFO) << "[CUSTOM KERNEL] Share outputs into ExecutionContext.";
      for (size_t i = 0; i < output_names.size(); ++i) {
        auto out_name = output_names[i];
        LOG(INFO) << "[CUSTOM KERNEL] output name - " << out_name;
        if (detail::IsDuplicableVar(out_name)) {
          PADDLE_ENFORCE(i == 0UL && output_names.size() == 1UL,
                         platform::errors::PreconditionNotMet(
                             "If custom operator's outputs contains
    `paddle::Vec("
                             ")` type, "
                             "it only can hold one output."));
          auto vec_true_outs = ctx.MultiOutput<Tensor>(out_name);
          PADDLE_ENFORCE_EQ(
              vec_true_outs.size(), outs.size(),
              platform::errors::InvalidArgument(
                  "The number of element in custom operator outputs is wrong, "
                  "expected contains %d Tensors, but actually contains %d "
                  "Tensors.",
                  vec_true_outs.size(), outs.size()));
          for (size_t j = 0; j < vec_true_outs.size(); ++j) {
            LOG(INFO) << "[CUSTOM KERNEL] moving output - " << j;
            experimental::MovesSharedStorage(
                std::dynamic_pointer_cast<pten::DenseTensor>(outs.at(j).impl())
                    .get(),
                vec_true_outs.at(j));
          }
        } else {
          LOG(INFO) << "[CUSTOM KERNEL] moving output - " << out_name;
          auto* true_out = ctx.Output<Tensor>(out_name);
          experimental::MovesSharedStorage(
              std::dynamic_pointer_cast<pten::DenseTensor>(outs.at(i).impl())
                  .get(),
              true_out);
        }
      }
    } catch (platform::EnforceNotMet& exception) {
      throw std::move(exception);
    } catch (std::exception& ex) {
      PADDLE_THROW(platform::errors::External("%s", ex.what()));
    } catch (...) {
      PADDLE_THROW(platform::errors::Fatal(
          "Custom kernel raises an unknown exception in rumtime."));
    }
    // refer to
    //
    https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/custom_operator.cc#L107
    */
}

// set pten::Kernel args_def_ from op_kernel_info
// because I can not set directly to pten::Kernel without exposing
// pten::KernelArgsDef to out
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
static void RunPtenKernelFunc(const OpKernelInfo& op_kernel_info,
                              pten::KernelContext* ctx,
                              const CustomKernelFunc& func) {
  VLOG(3) << "[CUSTOM KERNEL] RunPtenKernelFunc";

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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (op_kernel_info.GetBackend() == pten::Backend::GPU) {
    const pten::CUDAContext& dev_context =
        ctx->GetDeviceContext<pten::CUDAContext>();
    dev_ctx.set_stream(static_cast<void*>(dev_context.Stream().get()));
#endif
#ifdef PADDLE_WITH_ASCEND_CL
  } else if (op_kernel_info.GetBackend() == pten::Backend::NPU) {
    const pten::NPUContext& dev_context =
        ctx->GetDeviceContext<pten::NPUContext>();
    dev_ctx.set_stream(dev_context.stream());
#endif
  } else {
    LOG(ERROR) << "[CUSTOM KERNEL] mismatch kernel backend: "
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
  for (size_t i = 0; i < op_kernel_infos.size(); ++i) {
    auto& kernel_info = op_kernel_infos[i];
    auto op_type = OpKernelInfoHelper::GetOpName(kernel_info);
    auto kernel_key = OpKernelInfoHelper::GetKernelKey(kernel_info);

    VLOG(3) << "[CUSTOM KERNEL] registering [" << op_type << "]" << kernel_key;

    // 1.Check wether this kernel is valid for a specific operator
    if (!OpInfoMap::Instance().Has(op_type) &&
        !pten::KernelFactory::Instance().HasCompatiblePtenKernel(op_type)) {
      LOG(WARNING) << "[CUSTOM KERNEL] skipped: " << op_type
                   << " is not a valid operator.";
      continue;
    }

    // 2.Check wether this kernel should be registered to pten or fluid kernel
    // for pten kernel, meet the following three condition:
    // condition-0: FLAGS_run_pten_kernel is on
    // condition-1: op_type in pten::KernelFactory::Instance().kernels()
    // condition-2: kernel_key not in
    // pten::KernelFactory::Instance().kernels()[op_type]
    // others should be registered to fluid kernel
    bool register_to_pten = false;
    if (FLAGS_run_pten_kernel) {
      VLOG(3) << "[CUSTOM KERNEL] FLAGS_run_pten_kernel: "
              << FLAGS_run_pten_kernel;
      register_to_pten = true;
      if (!pten::KernelFactory::Instance().HasCompatiblePtenKernel(op_type)) {
        VLOG(3) << "[CUSTOM KERNEL] " << op_type
                << " has no compatible pten kernel.";
        register_to_pten = false;
      } else {
        VLOG(3) << "[CUSTOM KERNEL] " << op_type
                << " has compatible pten kernel.";
        if (pten::KernelFactory::Instance().kernels()[op_type].find(
                kernel_key) !=
            pten::KernelFactory::Instance().kernels()[op_type].end()) {
          LOG(WARNING) << "[CUSTOM KERNEL] skipped: " << kernel_key
                       << " has been registered already.";
          register_to_pten = false;
          continue;
        }
      }
    }
    VLOG(1) << "[CUSTOM KERNEL] register_to_pten: " << register_to_pten;

    if (register_to_pten) {
      VLOG(3) << "[CUSTOM KERNEL] try registering custom PTEN kernel";
      // pten::KernelFn
      auto& user_kernel_fn = OpKernelInfoHelper::GetKernelFn(kernel_info);
      pten::KernelFn kernel_fn = [kernel_info,
                                  user_kernel_fn](pten::KernelContext* ctx) {
        VLOG(3) << "[CUSTOM KERNEL] run custom PTEN kernel func in lambda.";
        RunPtenKernelFunc(kernel_info, ctx, user_kernel_fn);
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
    } else {
      VLOG(3) << "[CUSTOM KERNEL] try registering custom FLUID kernel";
      // trans pten kernelkey to org kernel key
      auto key = TransPtenKernelKeyToOpKernelType(kernel_key);
      if (OperatorWithKernel::AllOpKernels()[op_type].find(key) !=
          OperatorWithKernel::AllOpKernels()[op_type].end()) {
        LOG(WARNING) << "[CUSTOM KERNEL] skipped: " << kernel_key
                     << " has been registered already.";
        continue;
      }
      // register custom kernel to OperatorWithKernel::AllOpKernels()
      auto& user_kernel_fn = OpKernelInfoHelper::GetKernelFn(kernel_info);
      OperatorWithKernel::AllOpKernels()[op_type][key] = [user_kernel_fn](
          const framework::ExecutionContext& ctx) {
        VLOG(3) << "[CUSTOM KERNEL] run custom FLUID kernel func in lambda.";
        RunFluidKernelFunc(ctx, user_kernel_fn);
      };
      VLOG(3) << "[CUSTOM KERNEL] registered custom FLUID kernel";
    }
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
