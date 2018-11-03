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

#include "paddle/fluid/framework/details/fused_broadcast_op_handle.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/details/broadcast_op_handle_test.h"

namespace paddle {
namespace framework {
namespace details {

struct TestFusedBroadcastOpHandle : TestBroadcastOpHandle {
  std::vector<std::string> out_varnames_;

  void InitFusedBroadcastOp(std::vector<size_t> input_scope_idxes) {
    // initialize scope and var
    for (size_t i = 0; i < place_list_.size(); ++i) {
      local_scopes_.push_back(&(g_scope_.NewScope()));
      Scope& local_scope = local_scopes_.back()->NewScope();
      *local_scopes_.back()
           ->Var(details::kLocalExecScopeName)
           ->GetMutable<Scope*>() = &local_scope;
      for (size_t j = 0; j < input_scope_idxes.size(); ++j) {
        local_scope.Var("out_var" + j);
        if (i == j) local_scope.Var("in_var" + j);
      }
      param_scopes_.emplace_back(&local_scope);
    }

    // create op handle node
    std::unique_ptr<ir::Node> n =
        ir::CreateNodeForTest("fused_broadcast", ir::Node::Type::kOperation);
    if (use_gpu_) {
#ifdef PADDLE_WITH_CUDA
      op_handle_.reset(new FusedBroadcastOpHandle(
          n.get(), local_scopes_, place_list_, nccl_ctxs_.get()));
#else
      PADDLE_THROW("CUDA is not supported.");
#endif
    } else {
#ifdef PADDLE_WITH_CUDA
      op_handle_.reset(new FusedBroadcastOpHandle(
          n.get(), local_scopes_, place_list_, nccl_ctxs_.get()));
#else
      op_handle_.reset(
          new FusedBroadcastOpHandle(n.get(), local_scopes_, place_list_));
#endif
    }

    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      // add input var handle
      std::unique_ptr<ir::Node> in_node =
          ir::CreateNodeForTest("in_node" + i, ir::Node::Type::kVariable);
      VarHandle* in_var_handle =
          new VarHandle(in_node.get(), 1, input_scope_idxes[i], "in_var" + i,
                        place_list_[input_scope_idxes[i]]);
      vars_.emplace_back(in_var_handle);
      op_handle_->AddInput(in_var_handle);

      // add output var handle
      for (size_t j = 0; j < place_list_.size(); ++j) {
        std::unique_ptr<ir::Node> out_node =
            ir::CreateNodeForTest("out_node" + i, ir::Node::Type::kVariable);
        VarHandle* out_var_handle =
            new VarHandle(out_node.get(), 2, j, "out_var" + i, place_list_[j]);
        vars_.emplace_back(out_var_handle);
        op_handle_->AddOutput(out_var_handle);
      }
    }
  }

  void TestFusedBroadcastLoDTensor(std::vector<size_t> input_scope_idxes) {
    std::vector<std::vector<float>> send_vec;
    f::LoD lod{{0, 10, 20}};
    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      const std::string varname("in_var" + i);
      float val_scalar = static_cast<float>(i);
      send_vec.push_back(
          InitLoDTensor(varname, input_scope_idxes[i], lod, val_scalar));
    }

    op_handle_->Run(false);

    WaitAll();
    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      const std::string& varname("out_var" + i);
      for (size_t j = 0; j < place_list_.size(); ++j) {
        LoDTensorEqual(varname, send_vec[i], lod, param_scopes_[j]);
      }
    }
  }

  void TestFusedBroadcastSelectedRows(std::vector<size_t> input_scope_idxes) {
    std::vector<std::vector<float>> send_vector;
    std::vector<int64_t> rows{0, 1, 2, 3, 3, 0, 14, 7, 3, 1,
                              2, 4, 6, 3, 1, 1, 1,  1, 3, 7};
    int height = static_cast<int>(kDims[0] * 2);
    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      const std::string varname("in_var" + i);
      float val_scalar = static_cast<float>(i);
      send_vector.push_back(InitSelectedRows(varname, input_scope_idxes[i],
                                             rows, height, val_scalar));
    }

    op_handle_->Run(false);

    WaitAll();
    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      const std::string& varname("out_var" + i);
      for (size_t j = 0; j < place_list_.size(); ++j) {
        SelectedRowsEqual(varname, input_scope_idxes[i], send_vector[i], rows,
                          height);
      }
    }
  }
};

TEST(FusedBroadcastTester, CPULodTensor) {
  TestFusedBroadcastOpHandle test_op;
  std::vector<size_t> input_scope_idxes = {0, 1};
  test_op.InitCtxOnGpu(false);
  test_op.InitFusedBroadcastOp(input_scope_idxes);
  test_op.TestFusedBroadcastLoDTensor(input_scope_idxes);
}

TEST(FusedBroadcastTester, CPUSelectedRows) {
  TestFusedBroadcastOpHandle test_op;
  std::vector<size_t> input_scope_idxes = {0, 1};
  test_op.InitCtxOnGpu(false);
  test_op.InitFusedBroadcastOp(input_scope_idxes);
  test_op.TestFusedBroadcastSelectedRows(input_scope_idxes);
}

#ifdef PADDLE_WITH_CUDA
TEST(FusedBroadcastTester, GPULodTensor) {
  TestFusedBroadcastOpHandle test_op;
  std::vector<size_t> input_scope_idxes = {0, 1};
  test_op.InitCtxOnGpu(true);
  test_op.InitFusedBroadcastOp(input_scope_idxes);
  test_op.TestFusedBroadcastLoDTensor(input_scope_idxes);
}

TEST(FusedBroadcastTester, GPUSelectedRows) {
  TestFusedBroadcastOpHandle test_op;
  std::vector<size_t> input_scope_idxes = {0, 1};
  test_op.InitCtxOnGpu(true);
  test_op.InitFusedBroadcastOp(input_scope_idxes);
  test_op.TestFusedBroadcastSelectedRows(input_scope_idxes);
}
#endif

}  // namespace details
}  // namespace framework
}  // namespace paddle
