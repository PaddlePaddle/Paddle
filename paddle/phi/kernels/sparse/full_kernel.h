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

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace phi {

template <typename T, typename Context>
<<<<<<< HEAD
void CooFullLikeKernel(const Context& dev_ctx,
=======
void FullLikeCooKernel(const Context& dev_ctx,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                       const SparseCooTensor& x,
                       const Scalar& val,
                       DataType dtype,
                       SparseCooTensor* out);

template <typename T, typename Context>
<<<<<<< HEAD
void CsrFullLikeKernel(const Context& dev_ctx,
=======
void FullLikeCsrKernel(const Context& dev_ctx,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                       const SparseCsrTensor& x,
                       const Scalar& val,
                       DataType dtype,
                       SparseCsrTensor* out);

}  // namespace phi
