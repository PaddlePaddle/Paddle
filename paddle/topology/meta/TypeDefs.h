/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <unordered_map>
#include <unordered_set>

namespace paddle {
namespace topology {
namespace meta {
template <typename T>
using Set = std::unordered_set<T>;
template <typename T1, typename T2>
using Map = std::unordered_map<T1, T2>;

}  // namespace meta
}  // namespace topology
}  // namespace paddle
