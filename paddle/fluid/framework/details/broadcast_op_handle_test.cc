//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/broadcast_op_handle_test.h"

namespace paddle {
namespace framework {
namespace details {

TEST(BroadcastTester, TestCPUBroadcastTestLodTensor) {
  TestBroadcastOpHandle test_op;
  size_t input_scope_idx = 0;
  test_op.InitCtxOnGpu(false);
  test_op.InitBroadcastOp(input_scope_idx);
  test_op.TestBroadcastLodTensor(input_scope_idx);
}

TEST(BroadcastTester, TestCPUBroadcastTestSelectedRows) {
  TestBroadcastOpHandle test_op;
  size_t input_scope_idx = 0;
  test_op.InitCtxOnGpu(false);
  test_op.InitBroadcastOp(input_scope_idx);
  test_op.TestBroadcastSelectedRows(input_scope_idx);
}

#ifdef PADDLE_WITH_CUDA
TEST(BroadcastTester, TestGPUBroadcastTestLodTensor) {
  TestBroadcastOpHandle test_op;
  size_t input_scope_idx = 0;
  test_op.InitCtxOnGpu(true);
  test_op.InitBroadcastOp(input_scope_idx);
  test_op.TestBroadcastLodTensor(input_scope_idx);
}

TEST(BroadcastTester, TestGPUBroadcastTestSelectedRows) {
  TestBroadcastOpHandle test_op;
  size_t input_scope_idx = 0;
  test_op.InitCtxOnGpu(true);
  test_op.InitBroadcastOp(input_scope_idx);
  test_op.TestBroadcastSelectedRows(input_scope_idx);
}
#endif

}  // namespace details
}  // namespace framework
}  // namespace paddle
