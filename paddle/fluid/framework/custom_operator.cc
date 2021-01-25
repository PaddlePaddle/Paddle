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

#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/fluid/extension/include/op_function.h"

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {

namespace detail {

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
          "Failed to load dynamic operator library, error code(%s).", errorno));
  return func;
}

}  // namespace detail

// custom op kernel define
template <typename Func>
static void CallKernelFunc(const framework::ExecutionContext& ctx,
                           Func&& func) {
  VLOG(0) << "start run in CallKernelFunc";
  auto* x = ctx.Input<Tensor>(detail::kCustomOpInputPrefix + std::to_string(0));
  PADDLE_ENFORCE_NOT_NULL(x, "input x is nullptr.");
  PADDLE_ENFORCE(x->IsInitialized(), "input x is not initialized.");

  VLOG(0) << "run func in CallKernelFunc";
  auto outs = func(*x);

  VLOG(0) << "share output in CallKernelFunc";
  auto true_outs = ctx.MultiOutput<Tensor>(detail::kCustomOpOutputPrefix);
  for (size_t i = 0; i < true_outs.size(); ++i) {
    (true_outs)[i]->ShareDataWith(outs.at(i));
  }
}

template <typename DataType, typename Func>
class CustomOpKernel : public framework::OpKernel<DataType> {
 public:
  explicit CustomOpKernel(Func func) : func_(func) {}

  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(0) << "run in compute.";
    CallKernelFunc(ctx, func_);
  }

 private:
  Func func_;
};

class CustomOperator : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    VLOG(0) << "Infer shape of custom operator.";
  }
};

class CustomOpMaker : public OpProtoAndCheckerMaker {
 public:
  explicit CustomOpMaker(size_t input_num) : input_num_(input_num) {}

  void Make() override {
    for (size_t i = 0; i < input_num_; ++i) {
      std::string name = detail::kCustomOpInputPrefix + std::to_string(i);
      AddInput(name, "The input of Custom operator.");
    }
    // only one output
    AddOutput(detail::kCustomOpOutputPrefix, "The output of Custom Operator.")
        .AsDuplicable();
    AddComment(R"DOC(Custom Operator.)DOC");
  }

 private:
  size_t input_num_;
};

void RegisterOperator(const std::string& name, size_t input_num) {
  OpInfo info;

  // Op
  info.creator_ = [](const std::string& type, const VariableNameMap& inputs,
                     const VariableNameMap& outputs,
                     const AttributeMap& attrs) {
    return new CustomOperator(type, inputs, outputs, attrs);
  };

  // OpMaker
  info.proto_ = new proto::OpProto;
  info.checker_ = new OpAttrChecker();
  VLOG(0) << "proto type: " << name;
  CustomOpMaker custom_maker(input_num);
  info.proto_->set_type(name);
  custom_maker(info.proto_, info.checker_);
  PADDLE_ENFORCE_EQ(
      info.proto_->IsInitialized(), true,
      platform::errors::PreconditionNotMet(
          "Fail to initialize %s's OpProto, because %s is not initialized.",
          name, info.proto_->InitializationErrorString()));

  // GradOpMaker
  OpInfoMap::Instance().Insert(name, info);
}

void RegisterOperatorKernel(
    const std::string& name,
    const std::function<std::vector<Tensor>(const Tensor&)>& func) {
  // 1. construct kernel func
  CustomOpKernel<float, std::function<std::vector<Tensor>(const Tensor&)>>
      custom_kernel(func);

  // 2. insert kernel
  std::string library_type = "CPU";
  std::string data_layout = "ANYLAYOUT";
  OpKernelType key(ToDataType(std::type_index(typeid(float))),
                   platform::CPUPlace());

  VLOG(0) << "op name in kernel: " << name;
  VLOG(0) << "op kernel key: " << key;
  OperatorWithKernel::AllOpKernels()[name][key] =
      [custom_kernel](const framework::ExecutionContext& ctx) {
        VLOG(0) << "run custom kernel func.";
        custom_kernel.Compute(ctx);
      };
}

// load op api
void LoadCustomOperator(const std::string& dso_name) {
  void* handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);

  typedef OpFunctionMap& get_op_func_map_t();
  get_op_func_map_t* get_op_func_map =
      detail::DynLoad<get_op_func_map_t>(handle, "PD_GetOpFunctionMap");
  auto& op_func_map = get_op_func_map();
  auto& op_funcs = op_func_map.map();

  VLOG(0) << "size of op funcs map: " << op_funcs.size();
  for (auto& pair : op_funcs) {
    VLOG(0) << "pair first - op name: " << pair.first;
    // 1. forward op
    auto forward_in_num = pair.second.forward_in_num();
    auto forward_func = pair.second.forward_func();
    std::function<std::vector<Tensor>(const Tensor&)> func = std::move(
        forward_func
            .UnWrap<std::function<std::vector<Tensor>(const Tensor&)>>());
    // 2. register op
    VLOG(0) << "forward_in_num: " << forward_in_num;
    RegisterOperator(pair.first, forward_in_num);
    // 3. register op kernel
    RegisterOperatorKernel(pair.first, func);
  }
}

}  // namespace framework
}  // namespace paddle
