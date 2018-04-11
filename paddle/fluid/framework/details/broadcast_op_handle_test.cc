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

#include "paddle/fluid/framework/details/broadcast_op_handle.h"
#include "gtest/gtest.h"

#include "paddle/fluid/platform/device_context.h"

namespace f = paddle::framework;
namespace p = paddle::platform;

// test data amount
const f::DDim kDims = {20, 20};

class BroadcastTester : public ::testing::Test {
 public:
  void SetUp() override {
    int count = p::GetCUDADeviceCount();
    if (count <= 1) {
      LOG(WARNING) << "Cannot test multi-gpu Broadcast, because the CUDA "
                      "device count is "
                   << count;
      exit(0);
    }
    for (int i = 0; i < count; ++i) {
      gpu_list_.emplace_back(p::CUDAPlace(i));
    }
    ctxs_ = new p::ContextMap(gpu_list_);
  }

  template <class T>
  void BroadcastInitOp(int gpu_id = 0) {
    for (size_t j = 0; j < gpu_list_.size(); ++j) {
      local_scope_.push_back(&g_scope_.NewScope());
      auto* out_var = local_scope_[j]->Var("out");
      out_var->GetMutable<T>();
    }
    auto* in_var = local_scope_[gpu_id]->Var("input");
    in_var->GetMutable<T>();

    bc_op_handle_ =
        new f::details::BroadcastOpHandle(local_scope_, gpu_list_, *ctxs_);

    f::details::VarHandle* in_var_handle = new f::details::VarHandle();
    in_var_handle->place_ = gpu_list_[gpu_id];
    in_var_handle->name_ = "input";
    in_var_handle->version_ = 1;
    in_var_handle->generated_op_ = nullptr;
    bc_op_handle_->AddInput(in_var_handle);

    for (size_t j = 0; j < gpu_list_.size(); ++j) {
      f::details::VarHandle* out_var_handle = new f::details::VarHandle();
      out_var_handle->place_ = gpu_list_[j];
      out_var_handle->name_ = "out";
      out_var_handle->version_ = 2;
      out_var_handle->generated_op_ = bc_op_handle_;
      bc_op_handle_->AddOutput(out_var_handle);
    }
  }
  void BroadcastDestroy() {
    delete ctxs_;
    for (auto in : bc_op_handle_->inputs_) {
      delete in;
    }
    for (auto out : bc_op_handle_->outputs_) {
      delete out;
    }
    delete bc_op_handle_;
  }

 public:
  f::Scope g_scope_;
  p::ContextMap* ctxs_;
  std::vector<f::Scope*> local_scope_;
  std::vector<p::Place> gpu_list_;
  f::details::BroadcastOpHandle* bc_op_handle_;
};

TEST_F(BroadcastTester, BroadcastTestLodTensor) {
  int gpu_id = 0;
  BroadcastInitOp<f::LoDTensor>(gpu_id);

  auto in_var = local_scope_[gpu_id]->Var("input");
  auto in_lod_tensor = in_var->GetMutable<f::LoDTensor>();
  in_lod_tensor->mutable_data<float>(kDims, gpu_list_[gpu_id]);

  std::vector<float> send_vector(f::product(kDims), gpu_id + 12);
  for (size_t k = 0; k < send_vector.size(); ++k) {
    send_vector[k] = k;
  }
  f::LoD lod{{0, 10, 20}};
  paddle::framework::TensorFromVector<float>(
      send_vector, *(ctxs_->DevCtx(gpu_list_[gpu_id])), in_lod_tensor);
  in_lod_tensor->set_lod(lod);
  bc_op_handle_->Run(false);

  ctxs_->WaitAll();

  p::CPUPlace cpu_place;
  for (size_t j = 0; j < gpu_list_.size(); ++j) {
    auto out_var = local_scope_[j]->Var("out");
    auto out_tensor = out_var->Get<f::LoDTensor>();
    PADDLE_ENFORCE_EQ(out_tensor.lod(), lod, "lod is not equal.");

    f::Tensor result_tensor;
    f::TensorCopy(out_tensor, cpu_place, *(ctxs_->DevCtx(j)), &result_tensor);
    float* ct = result_tensor.mutable_data<float>(cpu_place);

    for (int64_t j = 0; j < f::product(kDims); ++j) {
      ASSERT_NEAR(ct[j], send_vector[j], 1e-5);
    }
  }

  BroadcastDestroy();
}

TEST_F(BroadcastTester, BroadcastTestSelectedRows) {
  int gpu_id = 0;
  BroadcastInitOp<f::SelectedRows>(gpu_id);

  auto in_var = local_scope_[gpu_id]->Var("input");
  auto in_selected_rows = in_var->GetMutable<f::SelectedRows>();
  auto value = in_selected_rows->mutable_value();
  value->mutable_data<float>(kDims, gpu_list_[gpu_id]);
  int height = kDims[0] * 2;
  std::vector<int64_t> rows{0, 1, 2, 3, 3, 0, 14, 7, 3, 1,
                            2, 4, 6, 3, 1, 1, 1,  1, 3, 7};
  in_selected_rows->set_height(height);
  in_selected_rows->set_rows(rows);

  std::vector<float> send_vector(f::product(kDims));
  for (size_t k = 0; k < send_vector.size(); ++k) {
    send_vector[k] = k;
  }
  paddle::framework::TensorFromVector<float>(
      send_vector, *(ctxs_->DevCtx(gpu_list_[gpu_id])), value);

  bc_op_handle_->Run(false);

  ctxs_->WaitAll();

  p::CPUPlace cpu_place;
  for (size_t j = 0; j < gpu_list_.size(); ++j) {
    auto out_var = local_scope_[j]->Var("out");
    auto& out_select_rows = out_var->Get<f::SelectedRows>();
    auto rt = out_select_rows.value();

    PADDLE_ENFORCE_EQ(out_select_rows.height(), height, "height is not equal.");
    for (size_t k = 0; k < out_select_rows.rows().size(); ++k) {
      PADDLE_ENFORCE_EQ(out_select_rows.rows()[k], rows[k]);
    }

    f::Tensor result_tensor;
    f::TensorCopy(rt, cpu_place, *(ctxs_->DevCtx(j)), &result_tensor);
    float* ct = result_tensor.data<float>();

    for (int64_t j = 0; j < f::product(kDims); ++j) {
      ASSERT_NEAR(ct[j], send_vector[j], 1e-5);
    }
  }

  BroadcastDestroy();
}
