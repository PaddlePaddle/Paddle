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
#include "paddle/fluid/pir/serialize_deserialize/include/third_party.h"

namespace pir {

Json GetAttrJson(const YAML::Node &action);

Json GetTypeJson(const YAML::Node &action);

std::string GetTypeName(const YAML::Node &action);

std::string GetAttrName(const YAML::Node &action);

Json BuildAttrJsonPatch(const YAML::Node &action);

Json BuildTypeJsonPatch(const YAML::Node &action);

Json ParseOpPatches(const YAML::Node &root);

Json ParseAttrPatches(const YAML::Node &root);

Json ParseTypePatches(const YAML::Node &root);

/* Yaml file is set to be empty by default. It's only used for testing. */
Json YamlParser(const std::string &version, const std::string &yaml_file = "");

}  // namespace pir
