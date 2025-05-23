// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ap::paddle {

class AxprValueImplBase {
 public:
  AxprValueImplBase(const AxprValueImplBase&) = delete;
  AxprValueImplBase(AxprValueImplBase&&) = delete;
  virtual ~AxprValueImplBase() {}

 protected:
  AxprValueImplBase() = default;
};

struct AxprValue {
  std::shared_ptr<AxprValueImplBase> value;
};

AxprValue AxprValueNone();
AxprValue AxprValueFromBool(bool data);
AxprValue AxprValueFromInt(int64_t data);
AxprValue AxprValueFromFloat(double data);
AxprValue AxprValueFromStr(const std::string& data);
AxprValue AxprValueFromList(const std::vector<AxprValue>& data);
bool AxprValueIsNone(const AxprValue& axpr_value);
bool AxprValueIsBool(const AxprValue& axpr_value);
bool AxprValueIsInt(const AxprValue& axpr_value);
bool AxprValueIsFloat(const AxprValue& axpr_value);
bool AxprValueIsStr(const AxprValue& axpr_value);
bool AxprValueIsList(const AxprValue& axpr_value);
bool AxprValueToBool(const AxprValue& axpr_value);
int64_t AxprValueToInt(const AxprValue& axpr_value);
double AxprValueToFloat(const AxprValue& axpr_value);
std::string AxprValueToStr(const AxprValue& axpr_value);
std::vector<AxprValue> AxprValueToList(const AxprValue& axpr_value);
AxprValue AxprValueImportModule(const std::string& module_name);
AxprValue AxprValueType(const AxprValue& axpr_value);
AxprValue AxprValueGetAttr(const AxprValue& axpr_value,
                           const std::string& attr_name);
AxprValue AxprValueGetItem(const AxprValue& axpr_value,
                           const AxprValue& item_name);
AxprValue AxprValueCall(const AxprValue& axpr_value,
                        const std::vector<AxprValue>& args);
std::string AxprValueStr(const AxprValue& axpr_value);
int64_t AxprValueLen(const AxprValue& axpr_value);

}  // namespace ap::paddle
