// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/container/flat_hash_map.h>
#include <absl/strings/string_view.h>
#include <absl/types/variant.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pybind11 {
namespace detail {
template <typename Key,
          typename Value,
          typename Hash,
          typename Equal,
          typename Alloc>
struct type_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>>
    : map_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>,
                 Key,
                 Value> {};

template <>
struct type_caster<absl::string_view> : string_caster<absl::string_view, true> {
};
}  // namespace detail
}  // namespace pybind11

namespace cinn::pybind {

void BindRuntime(pybind11::module *m);
void BindCommon(pybind11::module *m);
void BindLang(pybind11::module *m);
void BindIr(pybind11::module *m);
void BindBackends(pybind11::module *m);
void BindPoly(pybind11::module *m);
void BindOptim(pybind11::module *m);
void BindPE(pybind11::module *m);
void BindFramework(pybind11::module *m);
void BindUtils(pybind11::module *m);
void BindSchedule(pybind11::module *m);

__attribute__((visibility("default"))) extern void BindCINN(
    pybind11::module *m);

}  // namespace cinn::pybind
