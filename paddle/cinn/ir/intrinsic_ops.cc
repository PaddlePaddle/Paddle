// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/common/enforce.h"
namespace cinn::ir {

const char* IntrinsicOp::type_info() const { return IrNode::type_info(); }

const Type& IntrinsicOp::GetInputType(int offset) const {
  PADDLE_ENFORCE_LT(
      offset,
      input_types_.size(),
      ::common::errors::InvalidArgument("offset %d is out of range", offset));
  return input_types_[offset];
}
const Type& IntrinsicOp::GetOutputType(int offset) const {
  PADDLE_ENFORCE_LT(
      offset,
      output_types_.size(),
      ::common::errors::InvalidArgument("offset %d is out of range", offset));
  return output_types_[offset];
}

void IntrinsicOp::Verify(llvm::ArrayRef<Type> input_types,
                         llvm::ArrayRef<Type> output_types) const {
  PADDLE_ENFORCE_EQ(input_types.size(),
                    input_types_.size(),
                    ::common::errors::InvalidArgument(
                        "input_types.size() != input_types_.size()"));
  PADDLE_ENFORCE_EQ(output_types.size(),
                    output_types_.size(),
                    ::common::errors::InvalidArgument(
                        "output_types.size() != output_types_.size()"));

  for (int i = 0; i < input_types.size(); i++) {
    PADDLE_ENFORCE_EQ(input_types[i],
                      input_types_[i],
                      ::common::errors::InvalidArgument(
                          "input_types[%d] != input_types_[%d]", i, i));
  }

  for (int i = 0; i < output_types.size(); i++) {
    PADDLE_ENFORCE_EQ(output_types[i],
                      output_types_[i],
                      ::common::errors::InvalidArgument(
                          "output_types[%d] != output_types_[%d]", i, i));
  }
}

void IntrinsicOp::Verify(llvm::ArrayRef<Expr> inputs) const {
  PADDLE_ENFORCE_EQ(inputs.size(),
                    input_types_.size(),
                    ::common::errors::InvalidArgument(
                        "inputs.size() != input_types_.size()"));
  for (int i = 0; i < inputs.size(); i++) {
    PADDLE_ENFORCE_EQ(inputs[i].type().IgnoreConst(),
                      input_types_[i].IgnoreConst(),
                      ::common::errors::InvalidArgument(
                          "inputs[%d].type() != input_types_[%d]", i, i));
  }
}

void IntrinsicOp::Verify(llvm::ArrayRef<Expr> inputs,
                         llvm::ArrayRef<Expr> outputs) const {
  llvm::SmallVector<Type, 4> input_types, output_types;
  for (auto& e : inputs) input_types.push_back(e.type());
  for (auto& e : outputs) output_types.push_back(e.type());
  Verify(input_types, output_types);
}

Expr intrinsics::BufferGetDataHandle::Make(Expr buffer) {
  auto* n = new BufferGetDataHandle;
  n->Verify({buffer});
  n->buffer = buffer;
  n->set_type(n->GetOutputType(0));
  return Expr(n);
}

Expr intrinsics::BufferGetDataConstHandle::Make(Expr buffer) {
  auto* n = new BufferGetDataConstHandle;
  n->Verify({buffer});
  n->buffer = buffer;
  n->set_type(n->GetOutputType(0));
  return Expr(n);
}

Expr intrinsics::PodValueToX::Make(Expr pod_value_ptr, const Type& type) {
  auto* n = new PodValueToX;
  n->AddOutputType(type);
  n->Verify({pod_value_ptr});
  n->pod_value_ptr = pod_value_ptr;
  n->set_type(n->GetOutputType(0));
  return Expr(n);
}

Expr intrinsics::BufferCreate::Make(Expr buffer) {
  auto* n = new BufferCreate;
  n->set_type(Void());
  n->buffer = buffer;
  n->Verify({n->buffer});
  return Expr(n);
}

Expr intrinsics::GetAddr::Make(Expr data) {
  auto* n = new GetAddr;
  n->set_type(data.type().PointerOf());
  n->data = data;
  n->input_types_ = {data.type()};
  n->output_types_ = {data.type().PointerOf()};
  return Expr(n);
}

Expr intrinsics::ArgsConstruct::Make(Var var, llvm::ArrayRef<Expr> args) {
  auto* n = new ArgsConstruct;
  PADDLE_ENFORCE_EQ(
      var->type().ElementOf(),
      type_of<cinn_pod_value_t>(),
      ::common::errors::InvalidArgument(
          "var->type().ElementOf() != type_of<cinn_pod_value_t>()"));
  PADDLE_ENFORCE_GE(
      var->type().lanes(),
      1,
      ::common::errors::InvalidArgument("var->type().lanes() < 1"));
  for (auto& arg : args) {
    PADDLE_ENFORCE_EQ(arg.type(),
                      type_of<cinn_pod_value_t*>(),
                      ::common::errors::InvalidArgument(
                          "arg.type() != type_of<cinn_pod_value_t*>()"));
    n->AddInputType(var->type());
    n->AddInputType(arg.type());
  }
  n->var = var;
  n->AddOutputType(type_of<cinn_pod_value_t*>());
  n->args.assign(args.begin(), args.end());
  return Expr(n);
}

Expr intrinsics::BuiltinIntrin::Make(const std::string& name,
                                     llvm::ArrayRef<Expr> args,
                                     llvm::Intrinsic::ID id,
                                     int64_t arg_nums,
                                     const Type& type) {
  auto* n = new BuiltinIntrin;
  n->name = name;
  n->args.assign(args.begin(), args.end());
  n->id = id;
  n->arg_nums = arg_nums;
  PADDLE_ENFORCE_EQ(!type.is_unk(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The type is unknown. Please provide a valid type."));
  n->type_ = type;

  return Expr(n);
}

}  // namespace cinn::ir
