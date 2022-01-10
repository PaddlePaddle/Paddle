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
#include "paddle/fluid/string/string_helper.h"
#include "paddle/pten/api/ext/op_kernel_api.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/kernel_registry.h"

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
static void RunFluidKernelFunc(const framework::ExecutionContext& ctx,
                               const KernelFunc& func) {
  std::cout << "Custom Kernel: Run ComputeFunc." << std::endl;

  auto input_names = ctx.InNameList();
  auto output_names = ctx.OutNameList();
  auto op_attrs = ctx.Attrs();

  LOG(INFO) << "Custom Kernel: inputs name size - " << input_names.size();
  LOG(INFO) << "Custom Kernel: outputs name size - " << output_names.size();
  LOG(INFO) << "Custom Kernel: attrs name size - " << op_attrs.size();

  // check func inputs / outputs before real execution
  // 强制用户按照框架内的参数个数和顺序的方式不容易稳定：比如内部kernel增减或修改参数
  // 这也是为何在fluid kernel体系 且
  // 外部用户的kernel函数为functional时，需要让用户在kernel注册的时候提供参数、属性、返回的原因
  // 外部用户注册时提供信息后可以确保映射关系 但
  // 用户针对kernel也需要指定这些信息太冗杂
  // pten方式下用户参照各方都严格遵守的api.yaml或内部同名Kernel的形式实现时，不存在这个问题。

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
      PADDLE_ENFORCE_NOT_NULL(x, platform::errors::NotFound(
                                     "Input tensor (%s) is nullptr.", in_name));
      PADDLE_ENFORCE_EQ(x->IsInitialized(), true,
                        platform::errors::InvalidArgument(
                            "Input tensor (%s) is not initialized.", in_name));
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
          experimental::MovesSharedStorage(
              std::dynamic_pointer_cast<pten::DenseTensor>(outs.at(j).impl())
                  .get(),
              vec_true_outs.at(j));
        }
      } else {
        LOG(INFO) << "Custom Kernel: moving output - " << out_name;
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
  // https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/custom_operator.cc#L107
}

// set pten::Kernel args_def_ from op_kernel_info
// because I can not set directly to pten::Kernel without exposing
// pten::KernelArgsDef to out
static void ParseArgs(const OpKernelInfo* op_kernel_info,
                      pten::KernelArgsDef* args_def) {
  auto& input_defs = op_kernel_info->input_defs();
  auto& output_defs = op_kernel_info->output_defs();
  auto& attribute_defs = op_kernel_info->attribute_defs();

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

// custom Pten kernel mapping and register
// op_kernel_info is needed ? info from KernelContext is not enough such as
// attribute type
// TODO(SJC): List tensor
static void RunPtenKernelFunc(const OpKernelInfo& op_kernel_info,
                              pten::KernelContext* ctx,
                              const PtenKernelFunc& func) {
  LOG(WARNING) << "[CUSTOM PTEN KERNEL] RunPtenKernelFunc";

  // CPUContext for now
  const pten::CPUContext& cpu_context =
      ctx->GetDeviceContext<pten::CPUContext>();
  const DevContext& dev_ctx = reinterpret_cast<const DevContext&>(cpu_context);

  // input attr output parameters' sizes
  size_t input_size = ctx->InputsSize();
  size_t output_size = ctx->OutputsSize();
  size_t attr_size = ctx->AttrsSize();

  // input: range??
  std::vector<paddle::experimental::Tensor> custom_ins;
  std::vector<std::vector<paddle::experimental::Tensor>> custom_vec_ins;
  for (size_t in_idx = 0; in_idx < input_size; ++in_idx) {
    // if is Tensor:
    paddle::experimental::Tensor custom_t;
    const std::pair<int, int> range = ctx->InputRangeAt(in_idx);
    auto ctx_tensor = ctx->InputAt<pten::DenseTensor>(range.first);
    custom_t.set_impl(std::make_unique<pten::DenseTensor>(ctx_tensor));
    custom_ins.push_back(custom_t);
    // if is list[Tensor]:
    // const std::pair<int, int> range = ctx->InputRangeAt(in_idx);
    // std::vector<Tensor> arg = std::move(
    //   ctx->MoveInputsBetween<Tensor>(range.first, range.second));
  }
  // attr:
  std::vector<paddle::any> custom_attrs;
  auto attribute_defs = op_kernel_info.attribute_defs();
  for (size_t attr_idx = 0; attr_idx < attr_size; ++attr_idx) {
    if (attribute_defs[attr_idx].type_index == std::type_index(typeid(int))) {
      int arg = ctx->AttrAt<int>(attr_idx);
      custom_attrs.push_back(arg);
    }
    // Other type
  }
  // output:
  std::vector<paddle::experimental::Tensor*> custom_outs;
  std::vector<std::vector<paddle::experimental::Tensor*>> custom_vec_outs;
  for (size_t out_idx = 0; out_idx < output_size; ++out_idx) {
    // if is Tensor:
    const std::pair<int, int> range = ctx->OutputRangeAt(out_idx);
    auto ctx_tensor =
        ctx->MutableSharedOutputAt<pten::DenseTensor>(range.first);
    paddle::experimental::Tensor custom_t;
    custom_t.set_impl(ctx_tensor);
    custom_outs.push_back(&custom_t);
    // if is list[Tensor]:
    // const std::pair<int, int> range = ctx->InputRangeAt(in_idx);
    // std::vector<Tensor> arg = std::move(
    //   ctx->MoveInputsBetween<Tensor>(range.first, range.second));
  }

  // call user function
  func(dev_ctx, custom_ins, custom_vec_ins, custom_attrs, &custom_outs,
       &custom_vec_outs);

  LOG(WARNING) << "[CUSTOM KERNEL] finished";
}

void RegisterKernelWithMetaInfo(
    const std::vector<OpKernelInfo>& op_meta_infos) {
  for (size_t i = 0; i < op_meta_infos.size(); ++i) {
    auto& meta_info = op_meta_infos[i];
    auto op_name = OpKernelInfoHelper::GetOpName(meta_info);

    // check op exists
    if (!OpInfoMap::Instance().Has(op_name)) {
      LOG(WARNING) << "Operator (" << op_name << ") not exsits.";
      return;
    }

    // get kernel key
    auto kernel_key = OpKernelInfoHelper::GetKernelKey(meta_info);
    LOG(WARNING) << "[CUSTOM KERNEL] custom kernel key: " << kernel_key;

    if (FLAGS_run_pten_kernel &&
        pten::KernelFactory::Instance().HasCompatiblePtenKernel(op_name)) {
      LOG(WARNING) << "[CUSTOM KERNEL] begin register custom PTEN kernel";

      std::string kernel_name(op_name);
      auto& pten_kernel_fn = OpKernelInfoHelper::GetPtenKernelFn(meta_info);
      // construct pten::KernelFn
      pten::KernelFn kernel_fn = [meta_info,
                                  pten_kernel_fn](pten::KernelContext* ctx) {
        LOG(WARNING)
            << "[CUSTOM KERNEL] run custom PTEN kernel func in lambda.";
        RunPtenKernelFunc(meta_info, ctx, pten_kernel_fn);
      };

      // variadic_kernel_fn
      void* variadic_kernel_fn =
          OpKernelInfoHelper::GetPtenVariadicKernelFn(meta_info);
      pten::Kernel kernel(kernel_fn, variadic_kernel_fn);

      // args_parse_fn
      auto args_parse_fn = OpKernelInfoHelper::GetPtenArgsParseFn(meta_info);
      args_parse_fn(const_cast<OpKernelInfo*>(
          &meta_info));  // const_cast should not be used
      ParseArgs(&meta_info, kernel.mutable_args_def());

      // args_def_fn
      auto args_def_fn = OpKernelInfoHelper::GetPtenArgsDefFn(meta_info);
      args_def_fn(const_cast<OpKernelInfo*>(
          &meta_info));  // const_cast should not be used

      // write custom pten kernel to KernelFactory
      pten::KernelFactory::Instance().kernels()[kernel_name][kernel_key] =
          kernel;
      LOG(WARNING) << "[CUSTOM KERNEL] registered custom PTEN kernel";

    } else {
      LOG(WARNING) << "[CUSTOM KERNEL] begin register custom PTEN kernel";

      auto& kernel_fn = OpKernelInfoHelper::GetKernelFn(meta_info);
      // trans pten kernelkey to org kernel key
      auto key = TransPtenKernelKeyToOpKernelType(kernel_key);
      OperatorWithKernel::AllOpKernels()[op_name][key] =
          [kernel_fn](const framework::ExecutionContext& ctx) {
            LOG(WARNING)
                << "[CUSTOM KERNEL] run custom FLUID kernel func in lambda.";
            RunFluidKernelFunc(ctx, kernel_fn);
          };
      LOG(WARNING) << "[CUSTOM KERNEL] registered custom FLUID kernel";
    }
  }
}

void RegisterKernelWithMetaInfoMap(
    const paddle::OpKernelInfoMap& op_kernel_info_map) {
  auto& kernel_info_map = op_kernel_info_map.GetMap();
  VLOG(1) << "Custom Kernel: size of op_kernel_info_map - "
          << kernel_info_map.size();

  // pair: {op_type, OpKernelInfo}
  for (auto& pair : kernel_info_map) {
    VLOG(1) << "Custom Operator: pair first -> op name: " << pair.first;
    RegisterKernelWithMetaInfo(pair.second);
  }
}

////////////////////// User APIs ///////////////////////

// load op api
void LoadCustomKernel(const std::string& dso_name) {
  void* handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);
  VLOG(1) << "load custom_op lib: " << dso_name;

  typedef OpKernelInfoMap& get_op_kernel_info_map_t();
  auto* get_op_kernel_info_map = detail::DynLoad<get_op_kernel_info_map_t>(
      handle, "PD_GetOpKernelInfoMap");
  auto& op_kernel_info_map = get_op_kernel_info_map();

  RegisterKernelWithMetaInfoMap(op_kernel_info_map);
}

}  // namespace framework
}  // namespace paddle
