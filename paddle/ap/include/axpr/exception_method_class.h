// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/value.h"

namespace ap::axpr {

ADT_DEFINE_TAG(tWrapErrorAsValue);

using Exception = tWrapErrorAsValue<adt::errors::Error>;

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetExceptionClass();

template <typename ExceptionImpl>
adt::Result<axpr::Value> ConstructException(
    const axpr::Value&, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(msg, args.at(0).template CastTo<std::string>());
  adt::errors::Error error{ExceptionImpl{msg}};
  return GetExceptionClass().New(Exception{error});
}

template <typename ExceptionImpl, typename YieldT>
void YieldExceptionConstructor(const YieldT& Yield) {
  Yield(ExceptionImpl{}.class_name(), &ConstructException<ExceptionImpl>);
}

template <typename YieldT>
void ForEachExceptionConstructor(const YieldT& Yield) {
  YieldExceptionConstructor<adt::errors::RuntimeError>(Yield);
  YieldExceptionConstructor<adt::errors::InvalidArgumentError>(Yield);
  YieldExceptionConstructor<adt::errors::AttributeError>(Yield);
  YieldExceptionConstructor<adt::errors::NameError>(Yield);
  YieldExceptionConstructor<adt::errors::ValueError>(Yield);
  YieldExceptionConstructor<adt::errors::ZeroDivisionError>(Yield);
  YieldExceptionConstructor<adt::errors::TypeError>(Yield);
  YieldExceptionConstructor<adt::errors::IndexError>(Yield);
  YieldExceptionConstructor<adt::errors::KeyError>(Yield);
  YieldExceptionConstructor<adt::errors::MismatchError>(Yield);
  YieldExceptionConstructor<adt::errors::NotImplementedError>(Yield);
  YieldExceptionConstructor<adt::errors::SyntaxError>(Yield);
  YieldExceptionConstructor<adt::errors::ModuleNotFoundError>(Yield);
  YieldExceptionConstructor<adt::errors::AssertionError>(Yield);
}

}  // namespace ap::axpr
