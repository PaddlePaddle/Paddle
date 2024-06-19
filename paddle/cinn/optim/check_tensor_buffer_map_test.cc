// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/optim/check_tensor_buffer_map.h"

namespace cinn {
namespace optim {

TEST(CheckTensorBufferMap, not_equal) {
  using namespace ir;  // NOLINT
  auto B1 = _Buffer_::Make("B1", {Expr(100)});
  auto B2 = _Buffer_::Make("B2", {Expr(100)});
  auto T1 = _Tensor_::Make("A", Float16(), {Expr(100)}, {Expr(100)});
  auto T2 = _Tensor_::Make("A", Float16(), {Expr(100)}, {Expr(100)});
  T1->buffer = B1;
  T2->buffer = B2;

  auto S1 = Add::Make(Expr(T1), Expr(T2));

  PADDLE_ENFORCE_EQ(
      CheckTensorBufferMap(S1),
      false,
      phi::errors::InvalidArgument("CheckTensorBufferMap failed to detect "
                                   "tensor-buffer map with error."));
}

TEST(CheckTensorBufferMap, equal) {
  using namespace ir;  // NOLINT
  auto B1 = _Buffer_::Make("B1", {Expr(100)});
  auto T1 = _Tensor_::Make("A", Float16(), {Expr(100)}, {Expr(100)});
  auto T2 = _Tensor_::Make("A", Float16(), {Expr(100)}, {Expr(100)});
  T1->buffer = B1;
  T2->buffer = T1->buffer;

  auto S1 = Add::Make(Expr(T1), Expr(T2));

  PADDLE_ENFORCE_EQ(CheckTensorBufferMap(S1),
                    true,
                    phi::errors::InvalidArgument(
                        "CheckTensorBufferMap detected tensor-buffer map error "
                        "in an correct Expr."));
}

}  // namespace optim
}  // namespace cinn
