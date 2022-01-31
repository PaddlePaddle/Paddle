/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {

struct GridSizes {
  int64_t h;
  int64_t w;
  int64_t bs;
  int64_t coeffs_chans;
  int64_t gd;
  int64_t gh;
  int64_t gw;
  int64_t input_chans;
};

}  // namespace operators
}  // namespace paddle
