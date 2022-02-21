/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/backend.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"

namespace paddle {
namespace experimental {
namespace strings {

PADDLE_API Tensor empty(const ScalarArray& shape, Backend place = Backend::CPU);

PADDLE_API Tensor empty_like(const Tensor& x,
                             Backend place = Backend::UNDEFINED);

PADDLE_API Tensor lower(const Tensor& x, const std::string& encoding);

PADDLE_API Tensor upper(const Tensor& x, const std::string& encoding);

}  // namespace strings
}  // namespace experimental
}  // namespace paddle
