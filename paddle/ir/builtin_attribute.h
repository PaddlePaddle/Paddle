// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ir/attribute.h"
#include "paddle/ir/builtin_attribute_storage.h"

namespace ir {
///
/// \brief This macro is used to get a list of all built-in attributes in this
/// file. The built-in Dialect will use this macro to quickly register all
/// built-in attributes.
///
#define GET_BUILT_IN_ATTRIBUTE_LIST ir::StrAttribute

///
/// \brief Define built-in parameterless attributes. Please add the necessary
/// interface functions for built-in attributes through the macro
/// DECLARE_ATTRIBUTE_UTILITY_FUNCTOR.
///
/// NOTE(zhangbo9674): If you need to directly
/// cache the object of this built-in attribute in IrContext, please overload
/// the get method, and construct and cache the object in IrContext. For the
/// specific implementation method, please refer to StrAttribute.
///
/// The built-in attribute object get method is as follows:
/// \code{cpp}
///   ir::IrContext *ctx = ir::IrContext::Instance();
///   Attribute bool_attr = StrAttribute::get(ctx);
/// \endcode
///
class StrAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(StrAttribute, StrAttributeStorage);

  static StrAttribute get(ir::IrContext *ctx, const std::string &data) {
    return ir::AttributeManager::template get<StrAttribute>(
        ctx, const_cast<char *>(data.c_str()), data.size());
  }

  std::string data() const;

  const uint32_t &size() const;
};

}  // namespace ir
