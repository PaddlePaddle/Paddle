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

#include <string>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {

class CustomOpMaker : public OpProtoAndCheckerMaker {
 public:
  explicit CustomOpMaker(size_t input_num) : input_num_(input_num) {}

  void Make() override {
    for (size_t i = 0; i < input_num_; ++i) {
      std::string name = detail::kCustomOpInputPrefix + std::to_string(i);
      AddInput(name, "The input of Custom operator.");
    }
    // only one output
    AddOutput(detail::kCustomOpOutputPrefix, "The output of Custom Operator.");
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
  std::cout << "proto type: " << name << std::endl;
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

  std::cout << "op name in kernel: " << name << std::endl;
  std::cout << "op kernel key: " << key << std::endl;
  OperatorWithKernel::AllOpKernels()[name][key] =
      [custom_kernel](const framework::ExecutionContext& ctx) {
        VLOG(0) << "run custom kernel func.";
        custom_kernel.Compute(ctx);
      };
}

// load op api
void LoadCustomOperator(const std::string& dso_name) {
  void* handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);

  typedef OpFunctionHolder& get_op_func_map_t();
  get_op_func_map_t* get_op_func_holder =
      detail::DynLoad<get_op_func_map_t>(handle, "PD_GetOpFunctionHolder");
  auto& op_func_holder = get_op_func_holder();
  auto* op_funcs = op_func_holder.mutable_map();

  std::cout << "size of op funcs map: " << op_funcs->size() << std::endl;
  for (auto& pair : *op_funcs) {
    std::cout << "pair first - op name: " << pair.first << std::endl;
    std::cout << "pair second - func size: "
              << pair.second.mutable_funcs()->size() << std::endl;
    for (auto& func_pair : *(pair.second.mutable_funcs())) {
      std::function<std::vector<Tensor>(const Tensor&)> func =
          boost::get<std::function<std::vector<Tensor>(const Tensor&)>>(
              func_pair.second);
      // 1. parse args num
      size_t num = detail::ArgNumParser(func);
      std::cout << "size of args: " << num << std::endl;
      // 2. register op
      RegisterOperator(pair.first, num);
      // 3. register op kernel
      RegisterOperatorKernel(pair.first, func);
    }
  }
}

// func run test
// Tensor in;
// float* p = in.mutable_data<float>(make_ddim({1, 2, 3}),
// platform::CPUPlace());
// for (int i = 0; i < in.numel(); ++i) {
//   p[i] = static_cast<float>(i);
// }
// auto out = func(in);

}  // namespace framework
}  // namespace paddle
