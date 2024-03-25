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
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <initializer_list>
#include "paddle/phi/backends/dynload/port.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/common/enforce.h"
#include "glog/logging.h"

using Json = nlohmann::json;
namespace pir{
    
    Json writeType(const pir::Type& type);
    Json writeAttr(const pir::Attribute& attr);

    template<typename T>
    Json serializeTypeToJson(const T& type);

    template<typename T>
    Json serializeAttrToJson(const T& attr);
    
}//namepsace pir