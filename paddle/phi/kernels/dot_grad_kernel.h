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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void DotGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   const DenseTensor& dout,
                   DenseTensor* dx,
                   DenseTensor* dy);

template <typename T, typename Context>
void DotDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
<<<<<<< HEAD
                         const DenseTensor& dout,
                         const paddle::optional<DenseTensor>& ddx_opt,
                         const paddle::optional<DenseTensor>& ddy_opt,
=======
                         const DenseTensor& ddx,
                         const DenseTensor& ddy,
                         const DenseTensor& dout,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         DenseTensor* dx,
                         DenseTensor* dy,
                         DenseTensor* ddout);

template <typename T, typename Context>
void DotTripleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
<<<<<<< HEAD
                         const DenseTensor& dout,
                         const paddle::optional<DenseTensor>& ddx,
                         const paddle::optional<DenseTensor>& ddy,
                         const paddle::optional<DenseTensor>& d_dx,
                         const paddle::optional<DenseTensor>& d_dy,
                         const paddle::optional<DenseTensor>& d_ddout,
=======
                         const DenseTensor& ddx,
                         const DenseTensor& ddy,
                         const DenseTensor& d_dx,
                         const DenseTensor& d_dy,
                         const DenseTensor& dout,
                         const DenseTensor& d_ddout,
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                         DenseTensor* d_x,
                         DenseTensor* d_y,
                         DenseTensor* d_ddx,
                         DenseTensor* d_ddy,
                         DenseTensor* d_dout);

}  // namespace phi
