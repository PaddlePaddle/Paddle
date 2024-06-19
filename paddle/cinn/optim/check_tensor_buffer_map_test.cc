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

  // static Tensor Make(const std::string& name,
  //                    Type dtype,
  //                    const std::vector<Expr>& shape,
  //                    const std::vector<Expr>& domain,
  //                    const std::vector<Var>& reduce_axis = {});

  auto S1 = Add::Make(Expr(T1), Expr(T2));

  //   Expr Store::Make(Expr tensor, Expr value, const std::vector<Expr>
  //   &indices) { CHECK(tensor.As<_Tensor_>()) << "tensor should be _Tensor_
  //   type"; auto node = make_shared<Store>(); node->tensor = tensor;
  //   node->value = value;
  //   node->indices =
  //       utils::GetCompitableStoreLoadIndices(tensor.as_tensor_ref(),
  //       indices);
  //   if (tensor->type() != Void()) {
  //     node->set_type(
  //         tensor->type().ElementOf().with_lanes(node->index().type().lanes()));
  //   }
  //   return Expr(node);
  // }

  bool flag = CheckTensorBufferMapImpl(S1);

  LOG(INFO) << "debug tensor buffer test result: \n" << flag;

  bool target = true;
  ASSERT_EQ(flag, target);
}

TEST(CheckTensorBufferMap, equal) {
  using namespace ir;  // NOLINT
  auto B1 = _Buffer_::Make("B1", {Expr(100)});
  auto T1 = _Tensor_::Make("A", Float16(), {Expr(100)}, {Expr(100)});
  auto T2 = _Tensor_::Make("A", Float16(), {Expr(100)}, {Expr(100)});
  T1->buffer = B1;
  T2->buffer = T1->buffer;

  auto S1 = Add::Make(Expr(T1), Expr(T2));
  LOG(INFO) << "T1 buffer address: " << &(T1->buffer) << "\n";
  LOG(INFO) << "T2 buffer address: " << &(T2->buffer) << "\n";

  bool flag = CheckTensorBufferMapImpl(S1);

  LOG(INFO) << "debug tensor buffer test result: \n" << flag;

  bool target = false;
  ASSERT_EQ(flag, target);
}

}  // namespace optim
}  // namespace cinn
