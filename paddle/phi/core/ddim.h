// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/common/ddim.h"

namespace paddle {
namespace framework {
using DDim = common::DDim;
}
}  // namespace paddle

namespace phi {
using DDim = common::DDim;
using common::arity;
using common::contain_unknown_dim;
using common::flatten_to_1d;
using common::flatten_to_2d;
using common::flatten_to_3d;
using common::make_ddim;
using common::product;
using common::slice_ddim;
using common::stride;
using common::stride_numel;
using common::vectorize;
}  // namespace phi
