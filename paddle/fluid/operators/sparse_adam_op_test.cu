// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/adam_op.h"
#include "paddle/fluid/operators/lookup_table_op.h"
#include "paddle/fluid/platform/device_context.h"

using namespace paddle::platform;   // NOLINT
using namespace paddle::framework;  // NOLINT

USE_OP(lookup_table);
USE_OP(lookup_table_grad);
USE_OP(adam);

//#define USE_RANDOM_SEED // NOLINT

template <typename T>
T *CreateTensor(Scope &scope, const std::string &name,  // NOLINT
                const DDim &dim, const Place &place) {
  auto *tensor = scope.Var(name)->GetMutable<LoDTensor>();
  tensor->Resize(dim);
  return tensor->mutable_data<T>(place);
}

template <typename T>
void FillRandom(T *data, int numel, T low, T high, const Place &place,
                int seed = 1) {
  std::uniform_real_distribution<T> dist(low, high);
#ifdef USE_RANDOM_SEED
  std::random_device rd;
  std::mt19937 gen(rd());
#else
  std::mt19937 gen(seed);
#endif

  if (is_gpu_place(place)) {
    std::vector<T> cpu_data(numel);
    for (int i = 0; i < numel; ++i) cpu_data[i] = dist(gen);
    cudaMemcpy(data, cpu_data.data(), numel * sizeof(T),
               cudaMemcpyHostToDevice);
  } else {
    for (int i = 0; i < numel; ++i) data[i] = dist(gen);
  }
}

std::vector<int64_t> RandomRows(int h0, int h1, int seed = 1) {
  std::uniform_int_distribution<int64_t> dist(0, h1 - 1);
#ifdef USE_RANDOM_SEED
  std::random_device rd;
  std::mt19937 gen(rd());
#else
  std::mt19937 gen(seed);
#endif
  std::vector<int64_t> ret(h0);
  for (int i = 0; i < h0; ++i) {
    ret[i] = dist(gen);
  }
  return ret;
}

template <typename T>
void SelectedRowsToTensor(const SelectedRows &sr, LoDTensor *tensor, int height,
                          const Place &place) {
  std::vector<int64_t> rows(sr.rows().begin(), sr.rows().end());
  int width = sr.value().dims()[1];
  tensor->Resize({height, width});
  T *data = tensor->mutable_data<T>(place);
  const T *sr_data = sr.value().data<T>();
  if (is_gpu_place(place)) {
    cudaMemset(data, 0, sizeof(T) * tensor->numel());
#ifdef __NVCC__
    std::cerr << "gpu" << std::endl;
    for (size_t i = 0; i < rows.size(); ++i) {
      thrust::transform(
          thrust::device_pointer_cast(const_cast<const T *>(data)) +
              rows[i] * width,
          thrust::device_pointer_cast(const_cast<const T *>(data)) +
              (rows[i] + 1) * width,
          thrust::device_pointer_cast(sr_data) + i * width,
          thrust::device_pointer_cast(data) + rows[i] * width,
          thrust::plus<T>());
    }
#endif
    // add kernel here
  } else {
    memset(data, 0, sizeof(T) * tensor->numel());

    for (size_t i = 0; i < rows.size(); ++i) {
      for (int j = 0; j < width; ++j) {
        data[rows[i] * width + j] += sr_data[i * width + j];
      }
    }
  }
}

void TestMain(bool use_sparse = true, bool use_cpu = true) {
  float beta1 = 0.7;
  float beta2 = 0.8;
  float epsilon = 1.0e-8f;
  int height = 256;
  int width = 128;
  // bool use_cpu = true;

  int selected_row_height = 600;

  using Type = float;

  Scope scope;

  std::unique_ptr<DeviceContext> dev_ctx;

  if (!use_cpu) {
    dev_ctx.reset(new CUDADeviceContext(CUDAPlace(0)));
  } else {
    dev_ctx.reset(new CPUDeviceContext(CPUPlace()));
  }

  DDim dims{height, width};

  int size = height * width;
  auto place = dev_ctx->GetPlace();

  auto *ids =
      CreateTensor<int64_t>(scope, "Ids", {selected_row_height, 1}, place);
  auto rows = RandomRows(selected_row_height, height, 100);
  if (is_gpu_place(place)) {
    cudaMemcpy(ids, rows.data(), rows.size() * sizeof(rows[0]),
               cudaMemcpyHostToDevice);
  } else {
    memcpy(ids, rows.data(), rows.size() * sizeof(rows[0]));
  }

  auto *param = CreateTensor<Type>(scope, "Param", dims, place);  // W is Param
  FillRandom<Type>(param, size, 0.0, 1.0, place, 1);

  if (use_sparse) {
    scope.Var(GradVarName("W"))->GetMutable<SelectedRows>();
  } else {
    scope.Var(GradVarName("W"))->GetMutable<LoDTensor>();
  }

  auto *dout = CreateTensor<Type>(scope, GradVarName("Out"),
                                  {selected_row_height, width}, place);
  FillRandom<Type>(dout, selected_row_height * width, 0.0, 1.0, place, 1000);
  /*
  auto &sr = *scope.Var("GradSelectedRows")->GetMutable<SelectedRows>();
  sr.mutable_value()->Resize({selected_row_height, width});
  auto *grad = sr.mutable_value()->mutable_data<Type>(place);
  sr.set_rows(Vector<int64_t>(RandomRows(selected_row_height, height, 100)));
  FillRandom<Type>(grad, selected_row_height*width, 0.0, 1.0, place, 2);

  auto *tensor = scope.Var("GradTensor")->GetMutable<LoDTensor>();
  SelectedRowsToTensor<Type>(sr, tensor, height, place);
  std::string grad_name = use_sparse ? "GradSelectedRows" : "GradTensor";
  */

  auto *lr = CreateTensor<Type>(scope, "LearningRate", {1}, place);
  FillRandom<Type>(lr, 1, 0.001, 0.0011, place, 3);

  auto *mom1 = CreateTensor<Type>(scope, "Moment1", dims, place);
  FillRandom<Type>(mom1, size, 0.0, 1.0, place, 4);

  auto *mom2 = CreateTensor<Type>(scope, "Moment2", dims, place);
  FillRandom<Type>(mom2, size, 0.0, 1.0, place, 5);

  auto *beta1pow = CreateTensor<Type>(scope, "Beta1Pow", {1}, place);
  FillRandom<Type>(beta1pow, 1, 0.5, 1.0, place, 6);

  auto *beta2pow = CreateTensor<Type>(scope, "Beta2Pow", {1}, place);
  FillRandom<Type>(beta2pow, 1, 0.5, 1.0, place, 7);

  OpDesc lookup_table_grad_op_desc(
      "lookup_table_grad", {{"Ids", {"Ids"}},
                            {"W", {"Param"}},
                            {GradVarName("Out"), {GradVarName("Out")}}},
      {{GradVarName("W"), {GradVarName("W")}}}, {{"is_sparse", use_sparse}});

  OpDesc adam_op_desc(
      "adam", {{"Param", {"Param"}},
               {"LearningRate", {"LearningRate"}},
               {"Grad", {GradVarName("W")}},
               {"Moment1", {"Moment1"}},
               {"Moment2", {"Moment2"}},
               {"Beta1Pow", {"Beta1Pow"}},
               {"Beta2Pow", {"Beta2Pow"}}},
      {{"ParamOut", {"Param"}},
       {"Moment1Out", {"Moment1"}},
       {"Moment2Out", {"Moment2"}}},
      {{"beta1", beta1}, {"beta2", beta2}, {"epsilon", epsilon}});

  auto lookup_table_grad_op = OpRegistry::CreateOp(lookup_table_grad_op_desc);
  auto adam_op = OpRegistry::CreateOp(adam_op_desc);

  for (int i = 0; i < 100; ++i) {
    lookup_table_grad_op->Run(scope, place);
    adam_op->Run(scope, place);
  }
  dev_ctx->Wait();

  Tensor mom1_ret, mom2_ret, param_ret;
  CPUPlace cpu_place;
  TensorCopy(scope.FindVar("Moment1")->Get<LoDTensor>(), cpu_place, &mom1_ret);
  TensorCopy(scope.FindVar("Moment2")->Get<LoDTensor>(), cpu_place, &mom2_ret);
  TensorCopy(scope.FindVar("Param")->Get<LoDTensor>(), cpu_place, &param_ret);

  auto mom1_data = mom1_ret.data<Type>();
  auto mom2_data = mom2_ret.data<Type>();
  auto param_data = param_ret.data<Type>();
  std::cerr << std::accumulate(mom1_data, mom1_data + size, 0.0f) / size
            << std::endl;
  std::cerr << std::accumulate(mom2_data, mom2_data + size, 0.0f) / size
            << std::endl;
  std::cerr << std::accumulate(param_data, param_data + size, 0.0f) / size
            << std::endl;

  std::cerr << "Ends" << std::endl;
}

TEST(AdamSparse, cpu_sparse) { TestMain(true, true); }
TEST(AdamSparse, cpu) { TestMain(false, true); }
TEST(AdamSparse, gpu_sparse) { TestMain(true, false); }
TEST(AdamSparse, gpu) { TestMain(false, false); }
