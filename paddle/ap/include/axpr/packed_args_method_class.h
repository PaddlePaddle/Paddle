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
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/packed_args.h"

namespace ap::axpr {

template <typename ValueT>
struct MethodClassImpl<ValueT, PackedArgs<ValueT>> {};

template <typename ValueT>
struct MethodClassImpl<ValueT, TypeImpl<PackedArgs<ValueT>>> {
  using This = MethodClassImpl<ValueT, TypeImpl<PackedArgs<ValueT>>>;
  using Self = TypeImpl<PackedArgs<ValueT>>;
  adt::Result<ValueT> Call(const Self& self) { return &This::Construct; }

  static adt::Result<ValueT> Construct(const ValueT&,
                                       const std::vector<ValueT>& args) {
    return This{}.Make(args);
  }

  adt::Result<ValueT> Make(const std::vector<ValueT>& args) {
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(positional_args,
                      TryGetImpl<adt::List<ValueT>>(args.at(0)));
    ADT_LET_CONST_REF(keyword_args_val,
                      TryGetImpl<adt::List<ValueT>>(args.at(1)));
    axpr::AttrMap<ValueT> keyword_args;
    for (const auto& pair_val : *keyword_args_val) {
      ADT_LET_CONST_REF(pair, TryGetImpl<adt::List<ValueT>>(pair_val));
      ADT_CHECK(pair->size() == 2);
      ADT_LET_CONST_REF(key, TryGetImpl<std::string>(pair->at(0)));
      keyword_args->Set(key, pair->at(1));
    }
    return PackedArgs<ValueT>{positional_args, keyword_args};
  }
};

}  // namespace ap::axpr
