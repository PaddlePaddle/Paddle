// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>

namespace paddle {

namespace framework {
class VarDesc;
}  // namespace framework

namespace jit {
static const char PDMODEL_SUFFIX[] = ".pdmodel";
static const char PDPARAMS_SUFFIX[] = ".pdiparams";
static const char PROPERTY_SUFFIX[] = ".meta";

namespace utils {
bool IsPersistable(framework::VarDesc* desc_ptr);

bool StartsWith(const std::string& str, const std::string& suffix);

bool EndsWith(const std::string& str, const std::string& suffix);

void ReplaceAll(std::string* str,
                const std::string& old_value,
                const std::string& new_value);

bool FileExists(const std::string& file_path);

const std::vector<std::pair<std::string, std::string>> PdmodelFilePaths(
    const std::string& path);

void InitKernelSignatureMap();

}  // namespace utils
}  // namespace jit
}  // namespace paddle
