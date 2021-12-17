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
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(run_pten_kernel);

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
      func, platform::errors::NotFound(
                "Failed to load dynamic operator library, error message(%s).",
                errorno));
  return func;
}

inline bool IsDuplicableVar(const std::string& var_name) {
  std::string suffix = kTensorVectorSuffix;
  return var_name.rfind(suffix) != std::string::npos;
}

}  // namespace detail

// custom op kernel call function define
static void RunOrgKernelFunc(const framework::ExecutionContext& ctx,
                          const KernelFunc& func) {
  std::cout << "Custom Kernel: Run ComputeFunc." << std::endl;

  auto input_names = ctx.InNameList();
  auto output_names = ctx.OutNameList();
  auto op_attrs = ctx.Attrs();

  LOG(INFO) << "Custom Kernel: inputs name size - " << input_names.size();
  LOG(INFO) << "Custom Kernel: outputs name size - " << output_names.size();
  LOG(INFO) << "Custom Kernel: attrs name size - " << op_attrs.size();

  // check func inputs / outputs before real execution

  std::vector<paddle::experimental::Tensor> custom_ins;
  std::vector<std::vector<paddle::experimental::Tensor>> custom_vec_ins;

  for (auto& in_name : input_names) {
    LOG(INFO) << "Custom Kernel: input name - " << in_name;
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
                   "The %d-th tensor in input vector<tensor> (%s) is nullptr.",
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
      PADDLE_ENFORCE_NOT_NULL(x, platform::errors::NotFound("Input tensor (%s) is nullptr.", in_name));
      PADDLE_ENFORCE_EQ(x->IsInitialized(), true, platform::errors::InvalidArgument("Input tensor (%s) is not initialized.", in_name));
      paddle::experimental::Tensor custom_in;
      custom_in.set_impl(std::move(experimental::MakePtenDenseTensor(*x)));
      custom_ins.emplace_back(custom_in);
    }
  }

  std::vector<paddle::any> custom_attrs;
  for (auto& op_attr : op_attrs) {
    auto attr_name = op_attr.first;
    LOG(INFO) << "Custom Kernel: attr name - " << attr_name;
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

  LOG(INFO) << "Custom Kernel: Run ComputeFunc.";
  try {
    auto outs = func(custom_ins, custom_vec_ins, custom_attrs);

    LOG(INFO) << "Custom Kernel: Share outputs into ExecutionContext.";
    for (size_t i = 0; i < output_names.size(); ++i) {
      auto out_name = output_names[i];
      LOG(INFO) << "Custom Kernel: output name - " << out_name;
      if (detail::IsDuplicableVar(out_name)) {
        PADDLE_ENFORCE(i == 0UL && output_names.size() == 1UL,
                       platform::errors::PreconditionNotMet(
                           "If custom operator's outputs contains `paddle::Vec("
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
          LOG(INFO) << "Custom Kernel: moving output - " << j;
          experimental::MovesSharedStorage(std::dynamic_pointer_cast<pten::DenseTensor>(outs.at(j).impl()).get(), vec_true_outs.at(j));
        }
      } else {
        LOG(INFO) << "Custom Kernel: moving output - " << out_name;
        auto* true_out = ctx.Output<Tensor>(out_name);
        experimental::MovesSharedStorage(std::dynamic_pointer_cast<pten::DenseTensor>(outs.at(i).impl()).get(), true_out);
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
  // refer to https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/custom_operator.cc#L107
}

static void RunPtenKernelFunc(const framework::ExecutionContext& ctx,
                              const KernelFunc& func) {
  // TBD
}

void RegisterKernelWithMetaInfo(
    const std::vector<OpKernelInfo>& op_meta_infos) {
  auto& base_op_meta = op_meta_infos.front();

  auto op_name = OpKernelInfoHelper::GetOpName(base_op_meta);
  auto& kernel_fn = OpKernelInfoHelper::GetKernelFn(base_op_meta);

  std::string library_type = "PLAIN";
  std::string data_layout = "ANYLAYOUT";

  // TODO (add Pten kernel register here)
  // if (FLAGS_run_pten_kernel && pten::KernelFactory::Instance().HasCompatiblePtenKernel(op_name)) {

  //   // construct meta_kernel_fn
  //   // meta_kernel_fn = [kernel_fn](const DeviceContext* dev_ctx) {
  //   //       std::cout << "Custom Kernel: run custom kernel func." << std::endl;
  //   //       RunOrgKernelFunc(dev_ctx, kernel_fn);
  //   // };
  //   static void __reg_pt_kernel_custom_ALL_LAYOUT(pten::Kernel*);

  //   static const ::pten::KernelRegistrar __reg_pt_kernel_custom(op_name.c_str(), 
  //       pten::Backend::CPU, pten::DataLayout::ANY,  pten::DataType::INT64, 
  //       pten::KernelArgsParseFunctor<decltype(&kernel_fn)>::Parse,
  //       &__reg_pt_kernel_custom_ALL_LAYOUT,
  //       pten::KernelImpl<decltype(&kernel_fn), &kernel_fn>::Compute,
  //       reinterpret_cast<void*>(&::pten::KernelImpl<decltype(&kernel_fn), &kernel_fn>::VariadicCompute);

  //   pten::KernelKey kernel_key(pten::Backend::CPU, pten::DataLayout::ANY, pten::DataType::INT64);
  //   pten::KernelFactory::Instance().kernels()[op_name][kernel_key] =         
  //       [kernel_fn](const framework::ExecutionContext& ctx) {
  //         std::cout << "Custom Kernel: run custom kernel func in lambda."
  //                   << std::endl;
  //         RunOrgKernelFunc(ctx, kernel_fn);
  //       };

  // } else {
  paddle::framework::OpKernelType key(
      paddle::framework::proto::VarType::INT64, paddle::platform::CPUPlace(),
      paddle::framework::StringToDataLayout(data_layout),
      paddle::framework::StringToLibraryType(library_type.c_str()));
  OperatorWithKernel::AllOpKernels()[op_name][key] =
      [kernel_fn](const framework::ExecutionContext& ctx) {
        std::cout << "Custom Kernel: run custom kernel func in lambda."
                  << std::endl;
        RunOrgKernelFunc(ctx, kernel_fn);
      };
  // }
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
