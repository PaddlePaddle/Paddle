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

#include <cstdint>
#include <utility>

#include "paddle/fluid/pir/drr/include/drr_match_context.h"
#include "paddle/fluid/pir/drr/src/match_context_impl.h"
#include "paddle/phi/common/data_type.h"

namespace paddle::drr {

MatchContext::MatchContext(std::shared_ptr<const MatchContextImpl> impl)
    : impl_(std::move(impl)) {}

const pir::Value& MatchContext::Tensor(const std::string& tensor_name) const {
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
template double MatchContext::Attr<double>(const std::string&) const;
template std::string MatchContext::Attr<std::string>(const std::string&) const;
template std::vector<int32_t> MatchContext::Attr<std::vector<int32_t>>(
    const std::string&) const;
template std::vector<int64_t> MatchContext::Attr<std::vector<int64_t>>(
    const std::string&) const;
template std::vector<float> MatchContext::Attr<std::vector<float>>(
    const std::string&) const;
template phi::DataType MatchContext::Attr<phi::DataType>(
    const std::string&) const;
template phi::Place MatchContext::Attr<phi::Place>(const std::string&) const;

}  // namespace paddle::drr
