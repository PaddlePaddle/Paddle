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

#include "paddle/fluid/framework/custom_kernel.h"  // todo
#include "paddle/fluid/framework/op_kernel_api_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/pten/api/ext/op_kernel_api.h"

namespace paddle {
namespace framework {

// // custom op kernel call function define
// static void RunOpKernelFunc(const framework::ExecutionContext& ctx,
//                           const OpKernelFunc& func) {
//   VLOG(1) << "Custom Operator: Run ComputeFunc.";
//   try {
//     func(reinterpret_cast<PD_ExecutionContext*>(const_cast<ExecutionContext*>(&ctx)));
//   } catch (platform::EnforceNotMet& exception) {
//     throw std::move(exception);
//   } catch (std::exception& ex) {
//     PADDLE_THROW(platform::errors::External("%s", ex.what()));
//   } catch (...) {
//     PADDLE_THROW(platform::errors::Fatal(
//         "Custom operator raises an unknown exception in rumtime."));
//   }
// }

// void CustomerKernelBuilder::ResigterCustomKernel(const OpKernelFunc& func) {
//   std::string op_type = name_;
//   std::string library_type = "CPU";
//   std::string data_layout = "ANYLAYOUT";
//   paddle::framework::OpKernelType
//   key(paddle::framework::proto::VarType::FP32, paddle::platform::CPUPlace(),
//                    paddle::framework::StringToDataLayout(data_layout),
//                    paddle::framework::StringToLibraryType(library_type.c_str()));
//   // paddle::framework::OperatorWithKernel::AllOpKernels()[op_type][key] =
//   builder->compute_function;
//   VLOG(1) << "Custom Kernel: op kernel key: " << key;
//   OperatorWithKernel::AllOpKernels()[op_type][key] =
//       [func](const framework::ExecutionContext& ctx) {
//         VLOG(1) << "Custom Kernel: run custom kernel func in lambda.";
//         RunOpKernelFunc(ctx, func);
//       };
// }

////////////////////// User APIs ///////////////////////

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
      func, platform::errors::NotFound(
                "Failed to load dynamic operator library, error message(%s).",
                errorno));
  return func;
}

}  // namespace detail

// custom op kernel call function define
static void RunOpKernelFunc(const framework::ExecutionContext& ctx,
                            const OpKernelFunc& func) {
  std::cout << "Custom Kernel: Run ComputeFunc." << std::endl;
  try {
    func(reinterpret_cast<PD_ExecutionContext*>(
        const_cast<ExecutionContext*>(&ctx)));
  } catch (platform::EnforceNotMet& exception) {
    throw std::move(exception);
  } catch (std::exception& ex) {
    PADDLE_THROW(platform::errors::External("%s", ex.what()));
  } catch (...) {
    PADDLE_THROW(platform::errors::Fatal(
        "Custom kernel raises an unknown exception in rumtime."));
  }
}

void RegisterKernelWithMetaInfo(
    const std::vector<OpKernelInfo>& op_meta_infos) {
  auto& base_op_meta = op_meta_infos.front();

  auto op_name = OpKernelInfoHelper::GetOpName(base_op_meta);
  auto& kernel_fn = OpKernelInfoHelper::GetKernelFn(base_op_meta);

  std::string library_type = "PLAIN";
  std::string data_layout = "ANYLAYOUT";
  paddle::framework::OpKernelType key(
      paddle::framework::proto::VarType::INT64, paddle::platform::CPUPlace(),
      paddle::framework::StringToDataLayout(data_layout),
      paddle::framework::StringToLibraryType(library_type.c_str()));
  OperatorWithKernel::AllOpKernels()[op_name][key] =
      [kernel_fn](const framework::ExecutionContext& ctx) {
        std::cout << "Custom Kernel: run custom kernel func in lambda."
                  << std::endl;
        RunOpKernelFunc(ctx, kernel_fn);
      };
}

void RegisterKernelWithMetaInfoMap(
    const paddle::OpKernelInfoMap& op_meta_info_map) {
  auto& meta_info_map = op_meta_info_map.GetMap();
  std::cout << "Custom Kernel: size of op meta info map - "
            << meta_info_map.size() << std::endl;
  // pair: {op_type, OpMetaInfo}
  for (auto& pair : meta_info_map) {
    std::cout << "Custom Kernel: pair first -> op name: " << pair.first
              << std::endl;
    RegisterKernelWithMetaInfo(pair.second);
  }
}

// load kernel api
// typedef void (*PDKernelInitFn)();
void LoadCustomKernel(const std::string& dso_name) {
  void* handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);
  std::cout << "load custom_op lib: " << dso_name << std::endl;
  typedef OpKernelInfoMap& get_op_meta_info_map_t();
  auto* get_op_meta_info_map =
      detail::DynLoad<get_op_meta_info_map_t>(handle, "PD_GetOpKernelInfoMap");
  auto& op_meta_info_map = get_op_meta_info_map();

  RegisterKernelWithMetaInfoMap(op_meta_info_map);
}

}  // namespace framework
}  // namespace paddle
