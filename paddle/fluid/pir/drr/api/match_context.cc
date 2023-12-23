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

#include "paddle/fluid/pir/drr/api/match_context.h"

#include <cstdint>

#include "paddle/fluid/pir/drr/ir_operation.h"
#include "paddle/fluid/pir/drr/match_context_impl.h"

namespace pir {
namespace drr {

MatchContext::MatchContext(std::shared_ptr<const MatchContextImpl> impl)
    : impl_(impl) {}

const TensorInterface& MatchContext::Tensor(
    const std::string& tensor_name) const {
  return impl_->Tensor(tensor_name);
}

template <typename T>
T MatchContext::Attr(const std::string& attr_name) const {
  return impl_->Attr<T>(attr_name);
}

template bool MatchContext::Attr<bool>(const std::string&) const;
template int32_t MatchContext::Attr<int32_t>(const std::string&) const;
template int64_t MatchContext::Attr<int64_t>(const std::string&) const;
template float MatchContext::Attr<float>(const std::string&) const;
template std::string MatchContext::Attr<std::string>(const std::string&) const;
template std::vector<int32_t> MatchContext::Attr<std::vector<int32_t>>(
    const std::string&) const;
template std::vector<int64_t> MatchContext::Attr<std::vector<int64_t>>(
    const std::string&) const;

}  // namespace drr
}  // namespace pir
