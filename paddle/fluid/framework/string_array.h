/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <codecvt>
#include <iostream>
#include <locale>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace framework {

using String = std::string;
using Strings = std::vector<std::string>;
using Vocab = std::unordered_map<std::wstring, std::int32_t>;

// Convert the std::string type to the std::string type.
bool ConvertStrToWstr(const std::string& src, std::wstring* res);
// Convert the std::wstring type to the std::string type.
void ConvertWstrToStr(const std::wstring& src, std::string* res);
// Normalization Form Canonical Decomposition.
void NFD(const std::string& s, std::string* ret);

// Write the data which is type of
// std::unordered_map<td::string, int32_t> to ostream.
void StringMapToStream(std::ostream& os,
                       const std::unordered_map<std::string, int32_t>& data);

// Read the data which is type of
// std::unordered_map<td::string, int32_t> from istream.
void StringMapFromStream(std::istream& is,
                         std::unordered_map<std::string, int32_t>* data);
}  // namespace framework
}  // namespace paddle
