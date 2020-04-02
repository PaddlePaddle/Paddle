/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {
using FeedFetchType = LoDTensor;
using FeedFetchList = std::vector<FeedFetchType>;
using FetchUnmergedList = std::vector<std::vector<FeedFetchType>>;
using FetchResultType = boost::variant<FeedFetchList, FetchUnmergedList>;

class FetchVar : public boost::variant<LoDTensor, LoDTensorArray> {
 private:
  using FetchVarBase = boost::variant<LoDTensor, LoDTensorArray>;

 public:
  FetchVar() = default;
  FetchVar(const LoDTensor &lod_tensor) : FetchVarBase(lod_tensor) {}  // NOLINT
  FetchVar(const LoDTensorArray &lod_tensor_array)                     // NOLINT
      : FetchVarBase(lod_tensor_array) {}                              // NOLINT
};

// using FetchVar = boost::variant<LoDTensor, LoDTensorArray>;
using FetchVarList = std::vector<FetchVar>;

struct DataIsLoDTensor : public boost::static_visitor<bool> {
  bool operator()(const LoDTensor &data) const { return true; }
  bool operator()(const LoDTensorArray &) const { return false; }
};

struct DataIsLoDTensorArray : public boost::static_visitor<bool> {
  bool operator()(const LoDTensor &) const { return false; }
  bool operator()(const LoDTensorArray &data) const { return true; }
};

inline bool data_is_lod_tensor(const FetchVar &data) {
  return boost::apply_visitor(DataIsLoDTensor(), data);
}

inline bool data_is_lod_tensor_array(const FetchVar &data) {
  return boost::apply_visitor(DataIsLoDTensorArray(), data);
}

static const char kFeedOpType[] = "feed";
static const char kFetchOpType[] = "fetch";

}  // namespace framework
}  // namespace paddle
