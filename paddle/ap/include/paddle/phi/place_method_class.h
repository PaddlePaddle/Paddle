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

#include "paddle/ap/include/axpr/method_class.h"
#include "paddle/ap/include/axpr/naive_class_ops.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/axpr/value.h"
#include "paddle/ap/include/paddle/phi/place.h"

namespace ap::paddle {

inline adt::Result<axpr::Value> PlaceToString(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_LET_CONST_REF(self, self_val.template CastTo<phi::Place>());
  ADT_CHECK(args.size() == 0);
  const auto& str = self.DebugString();
  return str;
}

inline axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetPlaceClass() {
  static auto cls(axpr::MakeBuiltinClass<axpr::Value>(
      "Place", [&](const auto& DoEach) { DoEach("__str__", &PlaceToString); }));
  return axpr::MakeGlobalNaiveClassOps<phi::Place>(cls);
}

inline adt::Result<axpr::Value> CreateUndefinedPlace(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  phi::Place place;
  return GetPlaceClass().New(place);
}

inline adt::Result<axpr::Value> CreateCPUPlace(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  phi::Place place = phi::CPUPlace();
  return GetPlaceClass().New(place);
}

inline adt::Result<axpr::Value> CreateGPUPlace(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(device_id, args.at(0).template TryGet<int64_t>());
  phi::Place place = phi::GPUPlace(device_id);
  return GetPlaceClass().New(place);
}

inline adt::Result<axpr::Value> CreateGPUPinnedPlace(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 0);
  phi::Place place = phi::GPUPinnedPlace();
  return GetPlaceClass().New(place);
}

inline adt::Result<axpr::Value> CreateXPUPlace(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(device_id, args.at(0).template TryGet<int64_t>());
  phi::Place place = phi::XPUPlace(device_id);
  return GetPlaceClass().New(place);
}

inline adt::Result<axpr::Value> CreateIPUPlace(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  ADT_CHECK(args.size() == 1);
  ADT_LET_CONST_REF(device_id, args.at(0).template TryGet<int64_t>());
  phi::Place place = phi::IPUPlace(device_id);
  return GetPlaceClass().New(place);
}

inline adt::Result<axpr::Value> CreateCustomPlace(
    const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
  std::optional<phi::Place> place;
  if (args.size() == 1) {
    ADT_LET_CONST_REF(dev_type, args.at(0).template TryGet<std::string>());
    place = phi::CustomPlace(dev_type);
  } else if (args.size() == 2) {
    ADT_LET_CONST_REF(dev_type, args.at(0).template TryGet<std::string>());
    ADT_LET_CONST_REF(device_id, args.at(1).template TryGet<int64_t>());
    place = phi::CustomPlace(dev_type, device_id);
  } else {
    return adt::errors::TypeError{std::string() +
                                  "CustomPlace() takes 1 or 2 arguments, but " +
                                  std::to_string(args.size()) + " were given"};
  }
  ADT_CHECK(place.has_value());
  return GetPlaceClass().New(place.value());
}

}  // namespace ap::paddle
