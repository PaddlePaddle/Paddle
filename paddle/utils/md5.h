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

// The file has been adapted from ulwanski md5 project
// Copyright (c) 2021 Marek Ulwański
// Licensed under the MIT License -
// https://github.com/ulwanski/md5/blob/master/LICENSE

#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <cstring>
#include <string>
namespace paddle {
std::string md5(std::string data);
std::string md5(const void* data, size_t len);
}  // namespace paddle
