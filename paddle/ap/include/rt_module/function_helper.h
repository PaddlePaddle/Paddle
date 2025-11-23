// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/rt_module/arg_value.h"
#include "paddle/ap/include/rt_module/function.h"

namespace ap::rt_module {

struct FunctionHelper {
  adt::Result<axpr::Value> Apply(const Function& function,
                                 const std::vector<axpr::Value>& args) {
    const auto& func_declare = function->func_declare;
    ADT_LET_CONST_REF(ret_val, GetDefaultVal(func_declare->ret_type));
    ADT_LET_CONST_REF(ret_ptr, GetAddrAsVoidPtr(ret_val));
    void* ret = ret_ptr;
    std::vector<void*> void_ptr_args;
    void_ptr_args.reserve(args.size());
    ADT_CHECK(func_declare->arg_types->size() == args.size())
        << adt::errors::TypeError{
               std::string() + func_declare->func_id + "() takes " +
               std::to_string(func_declare->arg_types->size()) +
               " arguments, but " + std::to_string(args.size()) +
               " were given"};
    for (size_t i = 0; i < args.size(); ++i) {
      const auto& arg_axpr_value = args.at(i);
      {
        // check arg type
        const auto& arg_type = func_declare->arg_types->at(i);
        ADT_LET_CONST_REF(arg_value,
                          CastToArgValue<axpr::Value>(arg_axpr_value));
        ADT_CHECK(arg_value.GetType() == arg_type) << adt::errors::TypeError{
            std::string() + "the argument " + std::to_string(i) + " of " +
            func_declare->func_id + "() should be a " + arg_type.Name() +
            "(not " + arg_value.GetType().Name() + ")"};
      }
      ADT_LET_CONST_REF(ptr, GetAddrAsVoidPtr(arg_axpr_value));
      void_ptr_args.emplace_back(ptr);
    }
    ADT_RETURN_IF_ERR(function->dl_function.Apply(ret, void_ptr_args.data()));
    return ret_val;
  }

  adt::Result<axpr::Value> GetDefaultVal(const ArgType& arg_type) {
    return arg_type.Match(
        [&](const axpr::DataType& data_type) -> adt::Result<axpr::Value> {
          ADT_LET_CONST_REF(data_value, GetDataTypeDefaultVal(data_type));
          return data_value;
        },
        [&](const axpr::PointerType& pointer_type) -> adt::Result<axpr::Value> {
          ADT_LET_CONST_REF(pointer_value,
                            GetPointerTypeDefaultVal(pointer_type));
          return pointer_value;
        });
  }

  adt::Result<axpr::DataValue> GetDataTypeDefaultVal(
      const axpr::DataType& data_type) {
    return data_type.Match(
        [&](const auto& impl) -> adt::Result<axpr::DataValue> {
          using T = typename std::decay_t<decltype(impl)>::type;
          T val{};
          return axpr::DataValue{val};
        });
  }

  adt::Result<axpr::PointerValue> GetPointerTypeDefaultVal(
      const axpr::PointerType& pointer_type) {
    return pointer_type.Match(
        [&](const auto& impl) -> adt::Result<axpr::PointerValue> {
          using T = typename std::decay_t<decltype(impl)>::type;
          T ptr = nullptr;
          return axpr::PointerValue{ptr};
        });
  }

  adt::Result<void*> GetAddrAsVoidPtr(const axpr::Value& arg_value) {
    return arg_value.Match(
        [&](const axpr::DataValue& data) -> adt::Result<void*> {
          return data.Match(
              [&](const adt::Undefined&) -> adt::Result<void*> {
                static_assert(
                    axpr::DataValue::IsMyAlternative<adt::Undefined>(), "");
                // adt::Undefined represents cpp void type, because we cannot
                // define a void typed value.
                return nullptr;
              },
              [&](const auto& impl) -> adt::Result<void*> {
                using T = std::decay_t<decltype(impl)>;
                return const_cast<T*>(&impl);
              });
        },
        [&](const axpr::PointerValue& ptr) -> adt::Result<void*> {
          return ptr.Match([&](const auto& impl) -> adt::Result<void*> {
            using T = std::decay_t<decltype(impl)>;
            return const_cast<T*>(&impl);
          });
        },
        [&](const auto&) -> adt::Result<void*> {
          return adt::errors::TypeError{
              std::string() +
              "only DataValue or PointerValue are supported as so function "
              "arguments (not " +
              axpr::GetTypeName(arg_value) + ")."};
        });
  }
};

}  // namespace ap::rt_module
