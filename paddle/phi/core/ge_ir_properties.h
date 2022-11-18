/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

namespace phi {

struct GraphEngineIRProperties {
  GraphEngineIRProperties() {}

  GraphEngineIRProperties(bool persistable, bool parameter)
      : is_persistable(persistable), is_parameter(parameter) {}

  bool IsPersistable() { return is_persistable; }

  void SetIsPersistable(bool persistable) { is_persistable = persistable; }

  bool IsParameter() { return is_parameter; }

  void SetIsParameter(bool parameter) { is_parameter = parameter; }

  bool is_persistable{false};
  bool is_parameter{false};
};

}  // namespace phi
