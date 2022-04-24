//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/utils/variant.h"

namespace phi {

class Place;

// NOTE: Add needed type in the future
using Attribute = paddle::variant<bool,
                                  int,
                                  int64_t,
                                  float,
                                  double,
                                  std::string,
                                  std::vector<bool>,
                                  std::vector<int>,
                                  std::vector<int64_t>,
                                  std::vector<float>,
                                  std::vector<double>,
                                  std::vector<std::string>,
                                  Scalar,
                                  std::vector<Scalar>,
                                  IntArray,
                                  DataType,
                                  DataLayout,
                                  Place>;

}  // namespace phi
