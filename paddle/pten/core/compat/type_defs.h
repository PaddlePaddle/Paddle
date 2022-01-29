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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/variant.hpp>

namespace egr {
class EagerTensor;
}
namespace paddle {
namespace framework {
// The order should be as same as framework.proto
// NOTE(xiongkun): we extract from framework/typedef.h to ensure we can transfer
// enforce.h
class BlockDesc;
using Attribute = boost::variant<boost::blank,
                                 int,
                                 float,
                                 std::string,
                                 std::vector<int>,
                                 std::vector<float>,
                                 std::vector<std::string>,
                                 bool,
                                 std::vector<bool>,
                                 BlockDesc*,
                                 int64_t,
                                 std::vector<BlockDesc*>,
                                 std::vector<int64_t>,
                                 std::vector<double>>;
using AttributeMap = std::unordered_map<std::string, Attribute>;
}  // namespace framework

namespace imperative {

class VariableWrapper;
class SavedVariableWrapperList;
class VarBase;
class OpBase;
class GradOpNode;
class Tracer;

using WeakNameVarBaseMap =
    std::map<std::string, std::vector<std::weak_ptr<VarBase>>>;

namespace details {
template <typename T>
struct NameVarMapTrait {};

template <>
struct NameVarMapTrait<VarBase> {
  using Type = std::map<std::string, std::vector<std::shared_ptr<VarBase>>>;
};

template <>
struct NameVarMapTrait<VariableWrapper> {
  using Type = std::map<std::string, SavedVariableWrapperList>;
};

template <>
struct NameVarMapTrait<egr::EagerTensor> {
  using Type =
      std::map<std::string, std::vector<std::shared_ptr<egr::EagerTensor>>>;
};

}  // namespace details

template <typename T>
using NameVarMap = typename details::NameVarMapTrait<T>::Type;

using NameVarBaseMap = NameVarMap<VarBase>;
using NameVariableWrapperMap = NameVarMap<VariableWrapper>;
using NameTensorMap = NameVarMap<egr::EagerTensor>;

using VariableWrapperList = std::vector<std::shared_ptr<VariableWrapper>>;

}  // namespace imperative
}  // namespace paddle
