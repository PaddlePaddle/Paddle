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

#include "paddle/cinn/backends/function_prototype.h"

#include <glog/raw_logging.h>

#include <iostream>

#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(verbose_function_register);

namespace cinn {
namespace backends {

bool FunctionProto::Match(const ir::Call *op) const {
  if (name != op->name) return false;
  if (ret_type != op->type()) return false;
  if (op->read_args.size() != readonly_arg_types.size()) return false;
  if (op->write_args.size() != mutable_arg_types.size()) return false;

  for (int i = 0; i < op->read_args.size(); i++) {
    if (op->read_args[i].type() != readonly_arg_types[i]) return false;
  }
  for (int i = 0; i < op->write_args.size(); i++) {
    if (op->write_args[i].type() != mutable_arg_types[i]) return false;
  }
  return true;
}

void FunctionProto::AssertMatch(const ir::Call *op) const {
  CHECK_EQ(name, op->name);
  CHECK_EQ(ret_type, op->type())
      << "function proto " << name << " check failed";
  CHECK_EQ(op->read_args.size(), readonly_arg_types.size())
      << "function proto " << name << " check failed";
  CHECK_EQ(op->write_args.size(), mutable_arg_types.size())
      << "function proto " << name << " check failed";

  auto get_type = [](Expr u) {
    if (u.as_tensor() || u.as_buffer()) {
      Type t = u.type();
      return t.set_cpp_handle();
    }
    return u.type();
  };
  for (int i = 0; i < op->read_args.size(); i++) {
    if (readonly_arg_types[i] == type_of<cinn_buffer_t *>()) {
      if (!op->read_args[i].as_tensor()) continue;
    } else {
      CHECK_EQ(get_type(op->read_args[i]), readonly_arg_types[i]);
    }
  }
  for (int i = 0; i < op->write_args.size(); i++) {
    if (mutable_arg_types[i] == type_of<cinn_buffer_t *>()) {
      if (!op->write_args[i].as_tensor()) continue;
    } else {
      CHECK_EQ(get_type(op->write_args[i]), mutable_arg_types[i]);
    }
  }
}

void FunctionProto::CheckValid() {
  if (ret_type.is_void()) {
    CHECK(!mutable_arg_types.empty())
        << "A void function should have at least one mutable argument to "
           "output something";
  } else {
    CHECK(mutable_arg_types.empty())
        << "A function with return should not have mutable argument";
  }
}

FunctionProto::shape_inference_t FunctionProto::ShapeFollowNthArgument(int n) {
  return [=](const std::vector<Expr> &args, int value_offset) {
    CHECK_LT(n, args.size());
    auto x = args[n].as_tensor();
    CHECK(x);
    return x->shape;
  };
}

FunctionProto::FunctionProto(const std::string &name,
                             const std::vector<Type> &readonly_arg_types,
                             const std::vector<Type> &mutable_arg_types,
                             Type ret_type,
                             FunctionProto::shape_inference_t shape_inference)
    : name(name),
      readonly_arg_types(readonly_arg_types),
      mutable_arg_types(mutable_arg_types),
      ret_type(ret_type),
      shape_inference(shape_inference) {
  CheckValid();
}

FunctionProto *FunctionProtoRegistry::Lookup(const std::string &name) {
  auto it = data_.find(name);
  if (it != data_.end()) {
    return it->second.get();
  }
  return nullptr;
}

FunctionProto *FunctionProtoRegistry::Register(absl::string_view name,
                                               FunctionProto *x) {
#ifdef CINN_WITH_DEBUG
  if (FLAGS_verbose_function_register) {
    RAW_LOG_INFO("Register function prototype  [%s]", name.data());
  }
#endif  // CINN_WITH_DEBUG
  data_.emplace(name, std::unique_ptr<FunctionProto>(x));
  return x;
}

std::string FunctionProtoRegistry::debug_string() const {
  std::stringstream ss;
  for (auto &item : data_) {
    ss << item.first << "\n";
  }
  return ss.str();
}
}  // namespace backends
}  // namespace cinn
