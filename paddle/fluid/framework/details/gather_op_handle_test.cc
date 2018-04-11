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

#include "paddle/fluid/framework/details/gather_op_handle.h"
#include "gtest/gtest.h"

#include "paddle/fluid/platform/device_context.h"

namespace f = paddle::framework;
namespace p = paddle::platform;

// test data amount
const f::DDim kDims = {20, 20};

class GatherTester : public ::testing::Test {
 public:
  void InitCtx(bool use_gpu) {
    if (use_gpu) {
#ifdef PADDLE_WITH_CUDA
      int count = p::GetCUDADeviceCount();
      if (count <= 1) {
        LOG(WARNING) << "Cannot test multi-gpu Gather, because the CUDA "
                        "device count is "
                     << count;
        exit(0);
      }
      for (int i = 0; i < count; ++i) {
        auto p = p::CUDAPlace(i);
        gpu_list_.push_back(p);
        ctxs_.emplace_back(new p::CUDADeviceContext(p));
      }
#else
      PADDLE_THROW("CUDA is not support.");
#endif
    } else {
      int count = 8;
      for (int i = 0; i < count; ++i) {
        auto p = p::CPUPlace();
        gpu_list_.push_back(p);
        ctxs_.emplace_back(new p::CPUDeviceContext(p));
      }
    }
  }

  template <class T>
  void InitGatherOp(int input_scope_idx) {
    for (size_t j = 0; j < gpu_list_.size(); ++j) {
      local_scope_.push_back(&g_scope_.NewScope());
      auto* out_var = local_scope_[j]->Var("input");
      out_var->GetMutable<T>();
    }
    auto* in_var = local_scope_[input_scope_idx]->Var("out");
    in_var->GetMutable<T>();

    gather_op_handle_ = new f::details::GatherOpHandle(local_scope_, gpu_list_);

    f::details::VarHandle* out_var_handle = new f::details::VarHandle();
    out_var_handle->place_ = gpu_list_[input_scope_idx];
    out_var_handle->name_ = "out";
    out_var_handle->version_ = 2;
    out_var_handle->scope_idx_ = input_scope_idx;
    out_var_handle->generated_op_ = gather_op_handle_;
    gather_op_handle_->AddOutput(out_var_handle);

    for (size_t j = 0; j < gpu_list_.size(); ++j) {
      gather_op_handle_->dev_ctxes_[gpu_list_[j]] = ctxs_[j];
      f::details::VarHandle* in_var_handle = new f::details::VarHandle();
      in_var_handle->place_ = gpu_list_[j];
      in_var_handle->name_ = "input";
      in_var_handle->version_ = 1;
      in_var_handle->scope_idx_ = j;
      in_var_handle->generated_op_ = nullptr;
      gather_op_handle_->AddInput(in_var_handle);
    }
  }
  void GatherOpDestroy() {
    for (auto in : gather_op_handle_->inputs_) {
      delete in;
    }
    for (auto out : gather_op_handle_->outputs_) {
      delete out;
    }
    delete gather_op_handle_;
    for (size_t j = 0; j < ctxs_.size(); ++j) {
      delete ctxs_[j];
    }
  }

  void WaitAll() {
    for (size_t j = 0; j < ctxs_.size(); ++j) {
      ctxs_[j]->Wait();
    }
  }

  void TestGatherLodTensor() {
    //    int input_scope_idx = 0;
    //    InitGatherOp<f::LoDTensor>(input_scope_idx);
    //
    //    auto in_var = local_scope_[input_scope_idx]->Var("input");
    //    auto in_lod_tensor = in_var->GetMutable<f::LoDTensor>();
    //    in_lod_tensor->mutable_data<float>(kDims, gpu_list_[input_scope_idx]);
    //
    //    std::vector<float> send_vector(f::product(kDims), input_scope_idx +
    //    12);
    //    for (size_t k = 0; k < send_vector.size(); ++k) {
    //      send_vector[k] = k;
    //    }
    //    f::LoD lod{{0, 10, 20}};
    //    paddle::framework::TensorFromVector<float>(
    //        send_vector, *(ctxs_[input_scope_idx]), in_lod_tensor);
    //    in_lod_tensor->set_lod(lod);
    //
    //    gather_op_handle_->Run(false);
    //
    //    WaitAll();
    //
    //    p::CPUPlace cpu_place;
    //    for (size_t j = 0; j < gpu_list_.size(); ++j) {
    //      auto out_var = local_scope_[j]->Var("out");
    //      auto out_tensor = out_var->Get<f::LoDTensor>();
    //      PADDLE_ENFORCE_EQ(out_tensor.lod(), lod, "lod is not equal.");
    //
    //      f::Tensor result_tensor;
    //      f::TensorCopy(out_tensor, cpu_place, *(ctxs_[j]), &result_tensor);
    //      float* ct = result_tensor.mutable_data<float>(cpu_place);
    //
    //      for (int64_t j = 0; j < f::product(kDims); ++j) {
    //        ASSERT_NEAR(ct[j], send_vector[j], 1e-5);
    //      }
    //    }
    //
    //    GatherOpDestroy();
  }

  void TestGatherSelectedRows() {
    int output_scope_idx = 0;
    InitGatherOp<f::SelectedRows>(output_scope_idx);

    int height = kDims[0] * 2;
    std::vector<int64_t> rows{0, 1, 2, 3, 3, 0, 14, 7, 3, 1,
                              2, 4, 6, 3, 1, 1, 1,  1, 3, 7};
    std::vector<float> send_vector(f::product(kDims));
    for (size_t k = 0; k < send_vector.size(); ++k) {
      send_vector[k] = k;
    }

    for (size_t input_scope_idx = 0; input_scope_idx < gpu_list_.size();
         ++input_scope_idx) {
      auto in_var = local_scope_[input_scope_idx]->Var("input");
      auto in_selected_rows = in_var->GetMutable<f::SelectedRows>();
      auto value = in_selected_rows->mutable_value();
      value->mutable_data<float>(kDims, gpu_list_[input_scope_idx]);

      in_selected_rows->set_height(height);
      in_selected_rows->set_rows(rows);

      paddle::framework::TensorFromVector<float>(
          send_vector, *(ctxs_[input_scope_idx]), value);
      value->Resize(kDims);
    }

    gather_op_handle_->Run(false);

    WaitAll();

    p::CPUPlace cpu_place;

    auto out_var = local_scope_[output_scope_idx]->Var("out");
    auto& out_select_rows = out_var->Get<f::SelectedRows>();
    auto rt = out_select_rows.value();

    PADDLE_ENFORCE_EQ(out_select_rows.height(), height, "height is not equal.");
    for (size_t k = 0; k < out_select_rows.rows().size(); ++k) {
      PADDLE_ENFORCE_EQ(out_select_rows.rows()[k], rows[k % rows.size()]);
    }

    f::Tensor result_tensor;
    f::TensorCopy(rt, cpu_place, *(ctxs_[output_scope_idx]), &result_tensor);
    float* ct = result_tensor.data<float>();

    for (int64_t j = 0; j < f::product(kDims); ++j) {
      ASSERT_NEAR(ct[j], send_vector[j % send_vector.size()], 1e-5);
    }

    GatherOpDestroy();
  }

 public:
  f::Scope g_scope_;
  std::vector<p::DeviceContext*> ctxs_;
  std::vector<f::Scope*> local_scope_;
  std::vector<p::Place> gpu_list_;
  f::details::GatherOpHandle* gather_op_handle_;
};

// TEST_F(GatherTester, TestCPUGatherTestLodTensor) {
//  InitCtx(false);
//  TestGatherLodTensor();
//}

TEST_F(GatherTester, TestCPUGatherTestSelectedRows) {
  InitCtx(false);
  TestGatherSelectedRows();
}

#ifdef PADDLE_WITH_CUDA
// TEST_F(GatherTester, TestGPUGatherTestLodTensor) {
//  InitCtx(true);
//  TestGatherLodTensor();
//}

TEST_F(GatherTester, TestGPUGatherTestSelectedRows) {
  InitCtx(true);
  TestGatherSelectedRows();
}
#endif
