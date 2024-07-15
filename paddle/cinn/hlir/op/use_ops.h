// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/macros.h"

CINN_USE_REGISTER(nn_ops)
CINN_USE_REGISTER(broadcast_ops)
CINN_USE_REGISTER(elementwise_ops)
CINN_USE_REGISTER(transform_ops)
CINN_USE_REGISTER(gather_nd_ops)
CINN_USE_REGISTER(sort_ops)
CINN_USE_REGISTER(argmin_ops)
CINN_USE_REGISTER(argmax_ops)
CINN_USE_REGISTER(reduce_ops)
CINN_USE_REGISTER(custom_call_op)
CINN_USE_REGISTER(repeat_ops)
CINN_USE_REGISTER(one_hot_ops)
CINN_USE_REGISTER(lookup_table_ops)
CINN_USE_REGISTER(reciprocal_ops)
CINN_USE_REGISTER(gaussian_random_ops)
CINN_USE_REGISTER(uniform_random_ops)
CINN_USE_REGISTER(randint_ops)
CINN_USE_REGISTER(cholesky_ops)
CINN_USE_REGISTER(triangular_solve_ops)
CINN_USE_REGISTER(bitcast_convert_ops)
CINN_USE_REGISTER(assert_true_ops)
