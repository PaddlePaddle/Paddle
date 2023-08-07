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

#include "paddle/ir/pattern_rewrite/drr/api/match_context.h"
#include "paddle/ir/pattern_rewrite/drr/match_context_impl.h"

namespace ir {
namespace drr {

const TensorInterface& MatchContext::Tensor(
    const std::string& tensor_name) const {
  return impl_->Tensor(tensor_name);
}

template <typename T>
const T& MatchContext::Attr(const std::string& attr_name) const {
  return impl->Attr<T>(attr_name);
}

template const bool& MatchContext::Attr<bool>(const std::string&) const;
template const int32_t& MatchContext::Attr<int32_t>(const std::string&) const;
template const int64_t& MatchContext::Attr<int64_t>(const std::string&) const;
template const float& MatchContext::Attr<float>(const std::string&) const;

}  // namespace drr
}  // namespace ir
