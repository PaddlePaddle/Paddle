/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <string>
#include <vector>
#include "paddle/framework/lod_tensor.h"

namespace paddle {
namespace framework {
using FeedFetchType = LoDTensor;
using FeedFetchList = std::vector<FeedFetchType>;

static const std::string kFeedOpType = "feed";
static const std::string kFetchOpType = "fetch";
}  // namespace framework
}  // namespace paddle
