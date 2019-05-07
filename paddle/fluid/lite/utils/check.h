// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#define CHECK_OR_FALSE(cond)               \
  if (!(cond)) {                           \
    LOG(ERROR) << #cond << " test error!"; \
    return false;                          \
  }
#define CHECK_EQ_OR_FALSE(a__, b__)                           \
  if ((a__) != (b__)) {                                       \
    LOG(ERROR) << #a__ << " == " << #b__ << " check failed!"; \
    LOG(ERROR) << a__ << " != " << b__;                       \
    return false;                                             \
  }

#define CHECK_GT_OR_FALSE(a__, b__)                          \
  if (!((a__) > (b__))) {                                    \
    LOG(ERROR) << #a__ << " > " << #b__ << " check failed!"; \
    LOG(ERROR) << a__ << " <= " << b__;                      \
    return false;                                            \
  }

#define CHECK_GE_OR_FALSE(a__, b__)                           \
  if (!((a__) >= (b__))) {                                    \
    LOG(ERROR) << #a__ << " >= " << #b__ << " check failed!"; \
    LOG(ERROR) << a__ << " < " << b__;                        \
    return false;                                             \
  }
