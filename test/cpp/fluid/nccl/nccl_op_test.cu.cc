/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/common/place.h"

USE_NO_KERNEL_OP(ncclInit);
USE_OP_ITSELF(ncclAllReduce);
USE_OP_ITSELF(ncclReduce);
USE_OP_ITSELF(ncclBcast);
PD_DECLARE_KERNEL(ncclAllReduce, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(ncclReduce, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(ncclBcast, GPU, ALL_LAYOUT);

namespace f = paddle::framework;
namespace p = paddle::platform;

// test data amount
const f::DDim kDims = {20, 20};

// nccl op common tester, init communicator.
class NCCLTester : public ::testing::Test {
 public:
  void SetUp() override {
    int count = p::GetGPUDeviceCount();
    if (count <= 0) {
      LOG(WARNING) << "Cannot test gpu nccl, because the CUDA device count is "
                   << count;
      exit(0);
    }
    for (int i = 0; i < count; ++i) {
      gpu_list_.emplace_back(i);
    }

    phi::CPUPlace cpu_place;
    f::InitDevices();
    pool_ptr_ = &p::DeviceContextPool::Instance();

    NCCLInitOp();
  }

  void NCCLInitOp() {
    phi::CPUPlace cpu_place;
    std::unique_ptr<f::OpDesc> op1(new f::OpDesc);

    op1->SetType("ncclInit");
    op1->SetInput("parallel_scopes", {"p_scopes"});
    op1->SetOutput("Communicator", {"comm"});

    auto *var = g_scope_.Var("comm");
    var->GetMutable<p::Communicator>();

    auto *scope_var = g_scope_.Var("p_scopes");
    auto *p_scopes = scope_var->GetMutable<std::vector<f::Scope *>>();
    (*p_scopes).resize(gpu_list_.size());

    auto op = f::OpRegistry::CreateOp(*op1);
    VLOG(1) << "invoke NCCLInitOp.";
    op->Run(g_scope_, cpu_place);
    VLOG(1) << "NCCLInitOp finished.";
  }

  int GetGPUData(int gpu_id) { return gpu_id + 42; }

  template <class T>
  void PerThreadProgram(int gpu_id, const f::OpDesc &op_desc, f::Scope *scope) {
    std::unique_lock<std::mutex> lk(mu_);
    const f::OpDesc *op1 = &op_desc;

    phi::GPUPlace place(gpu_id);
    const auto &ctx = pool_ptr_->Get(place);

    auto *send_tensor = scope->Var("st")->GetMutable<phi::DenseTensor>();
    auto *recv_tensor = scope->Var("rt")->GetMutable<phi::DenseTensor>();

    if (!send_tensor->numel()) {
      send_tensor->mutable_data<T>(kDims, place);

      std::vector<T> send_vector(common::product(kDims), GetGPUData(gpu_id));
      paddle::framework::TensorFromVector<T>(send_vector, *ctx, send_tensor);
      VLOG(1) << "Send Tensor filled with elements " << send_tensor->numel();
    }

    lk.unlock();

    PADDLE_ENFORCE_EQ(
        send_tensor->numel(),
        common::product(kDims),
        common::errors::InvalidArgument("Tensor numel not match!"));

    auto op = f::OpRegistry::CreateOp(*op1);

    VLOG(1) << "Device : " << gpu_id << " invoke " << op_desc.Type();
    VLOG(1) << " send_tensor : " << send_tensor->numel()
            << " recv_tensor : " << recv_tensor->numel();
    op->Run(*scope, place);
    VLOG(1) << "Device : " << gpu_id << " finished " << op_desc.Type();
  }

  void testNcclReduceOp();
  void testNcclAllReduceOp();
  void testNcclBcastOp();

 public:
  p::DeviceContextPool *pool_ptr_;
  f::Scope g_scope_;
  std::mutex mu_;
  std::vector<int> gpu_list_;
};

void NCCLTester::testNcclAllReduceOp() {
  std::unique_ptr<f::OpDesc> op2(new f::OpDesc);
  op2->SetType("ncclAllReduce");
  op2->SetInput("X", {"st"});
  op2->SetInput("Communicator", {"comm"});
  op2->SetOutput("Out", {"rt"});

  std::vector<f::Scope *> dev_scopes;

  std::vector<std::thread> ths;

  for (size_t i = 0; i < gpu_list_.size(); ++i) {
    dev_scopes.emplace_back(&g_scope_.NewScope());
    std::thread th(&NCCLTester::PerThreadProgram<float>,
                   this,
                   gpu_list_[i],
                   *op2.get(),
                   dev_scopes[i]);
    ths.emplace_back(std::move(th));
  }

  for (size_t i = 0; i < gpu_list_.size(); ++i) {
    ths[i].join();
  }

  float expected_result = 0.0;
  for (int gpu_id : gpu_list_) {
    expected_result = expected_result + GetGPUData(gpu_id);
  }

  for (size_t i = 0; i < dev_scopes.size(); ++i) {
    phi::CPUPlace cpu_place;
    phi::GPUPlace gpu_place(gpu_list_[i]);

    auto &recv_tensor = dev_scopes[i]->FindVar("rt")->Get<phi::DenseTensor>();
    auto *rt = recv_tensor.data<float>();
    auto *result_tensor =
        dev_scopes[i]->Var("ct")->GetMutable<phi::DenseTensor>();
    result_tensor->Resize(kDims);
    auto *ct = result_tensor->mutable_data<float>(cpu_place);

    auto *dev_ctx = static_cast<phi::GPUContext *>(pool_ptr_->Get(gpu_place));
    paddle::memory::Copy(cpu_place,
                         ct,
                         phi::GPUPlace(gpu_list_[i]),
                         rt,
                         recv_tensor.numel() * sizeof(float),
                         dev_ctx->stream());
    dev_ctx->Wait();

    for (int64_t j = 0; j < common::product(kDims); ++j) {
      ASSERT_NEAR(ct[j], expected_result, 1e-5);
    }
  }
}

void NCCLTester::testNcclReduceOp() {
  std::unique_ptr<f::OpDesc> op2(new f::OpDesc);
  const int kRoot = 0;
  op2->SetType("ncclReduce");
  op2->SetInput("X", {"st"});
  op2->SetInput("Communicator", {"comm"});
  op2->SetOutput("Out", {"rt"});
  op2->SetAttr("root", kRoot);

  std::vector<f::Scope *> dev_scopes;

  std::vector<std::thread> ths;

  for (size_t i = 0; i < gpu_list_.size(); ++i) {
    dev_scopes.emplace_back(&g_scope_.NewScope());
    std::thread th(&NCCLTester::PerThreadProgram<float>,
                   this,
                   gpu_list_[i],
                   *op2.get(),
                   dev_scopes[i]);
    ths.emplace_back(std::move(th));
  }

  for (size_t i = 0; i < gpu_list_.size(); ++i) {
    ths[i].join();
  }

  float expected_result = 0.0;
  for (int gpu_id : gpu_list_) {
    expected_result = expected_result + GetGPUData(gpu_id);
  }

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(gpu_list_[kRoot]);

  auto &recv_tensor = dev_scopes[kRoot]->FindVar("rt")->Get<phi::DenseTensor>();
  auto *rt = recv_tensor.data<float>();
  auto *result_tensor =
      dev_scopes[kRoot]->Var("ct")->GetMutable<phi::DenseTensor>();
  result_tensor->Resize(kDims);
  auto *ct = result_tensor->mutable_data<float>(cpu_place);

  auto *dev_ctx = static_cast<phi::GPUContext *>(pool_ptr_->Get(gpu_place));
  paddle::memory::Copy(cpu_place,
                       ct,
                       phi::GPUPlace(gpu_list_[kRoot]),
                       rt,
                       recv_tensor.numel() * sizeof(float),
                       dev_ctx->stream());
  dev_ctx->Wait();

  for (int64_t j = 0; j < common::product(kDims); ++j) {
    ASSERT_NEAR(ct[j], expected_result, 1e-5);
  }
}

void NCCLTester::testNcclBcastOp() {
  std::unique_ptr<f::OpDesc> op2(new f::OpDesc);
  const int kRoot = 0;
  op2->SetType("ncclBcast");
  op2->SetInput("X", {"st"});
  op2->SetInput("Communicator", {"comm"});
  op2->SetOutput("Out", {"rt"});
  op2->SetAttr("root", kRoot);

  std::vector<f::Scope *> dev_scopes;

  std::vector<std::thread> ths;

  for (size_t i = 0; i < gpu_list_.size(); ++i) {
    dev_scopes.emplace_back(&g_scope_.NewScope());
    std::thread th(&NCCLTester::PerThreadProgram<float>,
                   this,
                   gpu_list_[i],
                   *op2.get(),
                   dev_scopes[i]);
    ths.emplace_back(std::move(th));
  }

  for (size_t i = 0; i < gpu_list_.size(); ++i) {
    ths[i].join();
  }

  const int idx = gpu_list_.size() - 1;
  float result = GetGPUData(kRoot);

  phi::CPUPlace cpu_place;
  phi::GPUPlace gpu_place(gpu_list_[idx]);

  std::string rt_str = "rt";
  if (idx == kRoot) {
    rt_str = "st";
  }
  auto &recv_tensor = dev_scopes[idx]->FindVar(rt_str)->Get<phi::DenseTensor>();
  auto *rt = recv_tensor.data<float>();
  auto *result_tensor =
      dev_scopes[idx]->Var("ct")->GetMutable<phi::DenseTensor>();
  result_tensor->Resize(kDims);
  auto *ct = result_tensor->mutable_data<float>(cpu_place);

  auto *dev_ctx = static_cast<phi::GPUContext *>(pool_ptr_->Get(gpu_place));
  paddle::memory::Copy(cpu_place,
                       ct,
                       phi::GPUPlace(gpu_list_[idx]),
                       rt,
                       recv_tensor.numel() * sizeof(float),
                       dev_ctx->stream());
  dev_ctx->Wait();

  for (int64_t j = 0; j < common::product(kDims); ++j) {
    ASSERT_NEAR(ct[j], result, 1e-5);
  }
}

// ncclInitOp with desc
TEST_F(NCCLTester, ncclInitOp) {}

TEST_F(NCCLTester, ncclOp) {
  // Serial execution is required for the same nccl comm.

  testNcclReduceOp();

  testNcclAllReduceOp();

  testNcclBcastOp();
}
