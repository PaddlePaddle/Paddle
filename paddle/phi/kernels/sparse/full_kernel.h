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
void FullLikeCooKernel(const Context& dev_ctx,
=======
void CooFullLikeKernel(const Context& dev_ctx,
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                       const SparseCooTensor& x,
                       const Scalar& val,
                       DataType dtype,
                       SparseCooTensor* out);

template <typename T, typename Context>
<<<<<<< HEAD
void FullLikeCsrKernel(const Context& dev_ctx,
=======
void CsrFullLikeKernel(const Context& dev_ctx,
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                       const SparseCsrTensor& x,
                       const Scalar& val,
                       DataType dtype,
                       SparseCsrTensor* out);

}  // namespace phi
