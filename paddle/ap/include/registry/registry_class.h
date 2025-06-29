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

#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/method.h"
#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/registry/registry.h"
#include "paddle/ap/include/registry/registry_singleton.h"

namespace ap::registry {

template <typename ValueT>
adt::Result<ValueT> RegisterAbstractDrrPass(const ValueT&,
                                            const std::vector<ValueT>& args) {
  ADT_CHECK(args.size() == 3) << adt::errors::TypeError{
      std::string() + "'Registry.abstract_drr_pass()' takes 3 arguments. but " +
      std::to_string(args.size()) + " were given."};
  const auto& drr_name_val = args.at(0);
  ADT_LET_CONST_REF(drr_name, axpr::TryGetImpl<std::string>(drr_name_val))
      << adt::errors::TypeError{std::string() +
                                "argument 1 of 'Registry.abstract_drr_pass()' "
                                "should be string, but '" +
                                axpr::GetTypeName(drr_name_val) +
                                "' were given."};
  const auto& nice_val = args.at(1);
  ADT_LET_CONST_REF(nice, axpr::TryGetImpl<int64_t>(nice_val))
      << adt::errors::TypeError{std::string() +
                                "argument 2 of 'Registry.abstract_drr_pass()' "
                                "should be int, but '" +
                                axpr::GetTypeName(nice_val) + "' were given."};
  const auto& cls_val = args.at(2);
  ADT_LET_CONST_REF(
      type_impl,
      axpr::TryGetTypeImpl<axpr::TypeImpl<axpr::ClassInstance<ValueT>>>(
          cls_val))
      << adt::errors::TypeError{
             std::string() +
             "argument 3 of 'Registry.abstract_drr_pass()' should "
             "be non-builtin class, but '" +
             axpr::GetTypeName(cls_val) + "' were given."};
  AbstractDrrPassRegistryItem item{drr_name, nice, type_impl.class_attrs};
  RegistrySingleton::Add(item);
  return adt::Nothing{};
}

template <typename ValueT>
adt::Result<ValueT> RegisterClassicDrrPass(const ValueT&,
                                           const std::vector<ValueT>& args) {
  ADT_CHECK(args.size() == 3) << adt::errors::TypeError{
      std::string() + "'Registry.classic_drr_pass()' takes 3 arguments. but " +
      std::to_string(args.size()) + " were given."};
  const auto& drr_name_val = args.at(0);
  ADT_LET_CONST_REF(drr_name, axpr::TryGetImpl<std::string>(drr_name_val))
      << adt::errors::TypeError{std::string() +
                                "argument 1 of 'Registry.classic_drr_pass()' "
                                "should be string, but '" +
                                axpr::GetTypeName(drr_name_val) +
                                "' were given."};
  const auto& nice_val = args.at(1);
  ADT_LET_CONST_REF(nice, axpr::TryGetImpl<int64_t>(nice_val))
      << adt::errors::TypeError{std::string() +
                                "argument 2 of 'Registry.classic_drr_pass()' "
                                "should be int, but '" +
                                axpr::GetTypeName(nice_val) + "' were given."};
  const auto& cls_val = args.at(2);
  ADT_LET_CONST_REF(
      type_impl,
      axpr::TryGetTypeImpl<axpr::TypeImpl<axpr::ClassInstance<ValueT>>>(
          cls_val))
      << adt::errors::TypeError{
             std::string() +
             "argument 3 of 'Registry.classic_drr_pass()' should "
             "be non-builtin class, but '" +
             axpr::GetTypeName(cls_val) + "' were given."};
  ClassicDrrPassRegistryItem item{drr_name, nice, type_impl.class_attrs};
  RegistrySingleton::Add(item);
  return adt::Nothing{};
}

template <typename ValueT>
adt::Result<ValueT> RegisterAccessTopoDrrPass(const ValueT&,
                                              const std::vector<ValueT>& args) {
  ADT_CHECK(args.size() == 3) << adt::errors::TypeError{
      std::string() +
      "'Registry.access_topo_drr_pass()' takes 3 arguments. but " +
      std::to_string(args.size()) + " were given."};
  const auto& drr_name_val = args.at(0);
  ADT_LET_CONST_REF(drr_name, axpr::TryGetImpl<std::string>(drr_name_val))
      << adt::errors::TypeError{
             std::string() +
             "argument 1 of 'Registry.access_topo_drr_pass()' "
             "should be string, but '" +
             axpr::GetTypeName(drr_name_val) + "' were given."};
  ADT_LET_CONST_REF(pass_tag_name, axpr::TryGetImpl<std::string>(args.at(1)))
      << adt::errors::TypeError{
             std::string() +
             "argument 2 of 'Registry.access_topo_drr_pass()' "
             "should be int, but '" +
             axpr::GetTypeName(args.at(1)) + "' were given."};
  const auto& cls_val = args.at(2);
  ADT_LET_CONST_REF(
      type_impl,
      axpr::TryGetTypeImpl<axpr::TypeImpl<axpr::ClassInstance<ValueT>>>(
          cls_val))
      << adt::errors::TypeError{
             std::string() +
             "argument 3 of 'Registry.access_topo_drr_pass()' should "
             "be non-builtin class, but '" +
             axpr::GetTypeName(cls_val) + "' were given."};
  AccessTopoDrrPassRegistryItem item{
      drr_name, pass_tag_name, 0, type_impl.class_attrs};
  RegistrySingleton::Add(item);
  return adt::Nothing{};
}

template <typename ValueT>
axpr::TypeImpl<axpr::BuiltinClassInstance<ValueT>> MakeRegistryClass() {
  static auto cls(
      axpr::MakeBuiltinClass<ValueT>("Registry", [&](const auto& DoEach) {
        DoEach("abstract_drr_pass", &RegisterAbstractDrrPass<ValueT>);
        DoEach("classic_drr_pass", &RegisterClassicDrrPass<ValueT>);
        DoEach("access_topo_drr_pass", &RegisterAccessTopoDrrPass<ValueT>);
      }));
  using Self = Registry;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::registry
