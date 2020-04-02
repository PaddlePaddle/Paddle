// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/memory/memcpy.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;
namespace paddle {
namespace imperative {

void TensorAdd(const framework::Variable& src, framework::Variable* dst);

#if defined(PADDLE_WITH_CUDA)
template <typename T>
int TensorGPUAddTest(platform::CUDAPlace place, T t1, T t2) {
  framework::Variable var1;
  framework::Variable var2;
  std::vector<T> src_data(10, t1);
  std::vector<T> dst_data(10, t2);
  std::vector<T> result;
  platform::CPUPlace src_place;
  for (unsigned int i = 0; i < 10; i++) {
    result.emplace_back(src_data[i] + dst_data[i]);
  }
  std::vector<int64_t> dims = {2, 5};
  auto* src = var1.GetMutable<framework::LoDTensor>();
  auto* dst = var2.GetMutable<framework::LoDTensor>();
  src->Resize(framework::make_ddim(dims));
  dst->Resize(framework::make_ddim(dims));
  auto* src_mutable = src->mutable_data<T>(place);
  auto* dst_mutable = dst->mutable_data<T>(place);
  paddle::memory::Copy(place, src_mutable, src_place, src_data.data(),
                       sizeof(T) * src_data.size(), 0);
  paddle::memory::Copy(place, dst_mutable, src_place, dst_data.data(),
                       sizeof(T) * dst_data.size(), 0);
  imperative::TensorAdd(var1, &var2);
  framework::LoDTensor rlt;
  platform::CPUPlace rlt_place;
  framework::TensorCopySync(*dst, rlt_place, &rlt);

  for (unsigned int i = 0; i < rlt.numel(); i++) {
    if (rlt.data<T>()[i] != result[i]) return 1;
  }
  return 0;
}
#endif

template <typename T>
int TensorCPUAddTest(platform::CPUPlace place, T t1, T t2) {
  framework::Variable var1;
  framework::Variable var2;
  std::vector<T> src_data(10, t1);
  std::vector<T> dst_data(10, t2);
  std::vector<T> result;
  platform::CPUPlace src_place;
  for (unsigned int i = 0; i < 10; i++) {
    result.emplace_back(src_data[i] + dst_data[i]);
  }
  std::vector<int64_t> dims = {2, 5};
  auto* src = var1.GetMutable<framework::LoDTensor>();
  auto* dst = var2.GetMutable<framework::LoDTensor>();
  src->Resize(framework::make_ddim(dims));
  dst->Resize(framework::make_ddim(dims));
  auto* src_mutable = src->mutable_data<T>(place);
  auto* dst_mutable = dst->mutable_data<T>(place);
  paddle::memory::Copy(place, src_mutable, src_place, src_data.data(),
                       sizeof(T) * src_data.size());
  paddle::memory::Copy(place, dst_mutable, src_place, dst_data.data(),
                       sizeof(T) * dst_data.size());
  imperative::TensorAdd(var1, &var2);
  framework::LoDTensor rlt;
  platform::CPUPlace rlt_place;
  framework::TensorCopySync(*dst, rlt_place, &rlt);

  for (unsigned int i = 0; i < rlt.numel(); i++) {
    if (rlt.data<T>()[i] != result[i]) return 1;
  }
  return 0;
}

TEST(test_add_functor, add_functor) {
#if defined(PADDLE_WITH_CUDA)
  platform::CUDAPlace gpu_place(0);
#endif
  platform::CPUPlace cpu_place;

  int cpu_res = 1;
  cpu_res = TensorCPUAddTest(cpu_place, 1.0, 0.0);
  EXPECT_EQ(cpu_res, 0);
  cpu_res = TensorCPUAddTest(cpu_place, static_cast<double>(1.0),
                             static_cast<double>(2.0));
  EXPECT_EQ(cpu_res, 0);
#if defined(PADDLE_WITH_CUDA)
  int gpu_res = 1;
  gpu_res = TensorGPUAddTest(gpu_place, 1.0, 0.0);
  EXPECT_EQ(gpu_res, 0);
  gpu_res = TensorGPUAddTest(gpu_place, static_cast<double>(1.0),
                             static_cast<double>(2.0));
  EXPECT_EQ(gpu_res, 0);
#endif
}

static void CopyVar(const framework::Variable& var,
                    framework::Variable* dst_ptr) {
  auto& dst = *dst_ptr;
  dst.Clear();
  if (var.IsType<framework::LoDTensor>()) {
    const auto& src_tensor = var.Get<framework::LoDTensor>();
    auto* dst_tensor = dst.GetMutable<framework::LoDTensor>();
    framework::TensorCopySync(src_tensor, src_tensor.place(), dst_tensor);
  } else {
    const auto& src_selected_rows = var.Get<framework::SelectedRows>();
    auto* dst_selected_rows = dst.GetMutable<framework::SelectedRows>();
    dst_selected_rows->set_rows(src_selected_rows.rows());
    dst_selected_rows->set_height(src_selected_rows.height());
    framework::TensorCopySync(src_selected_rows.value(),
                              src_selected_rows.value().place(),
                              dst_selected_rows->mutable_value());
  }
}

static bool IsEqualVar(const framework::Variable& var1,
                       const framework::Variable& var2) {
  if (var1.Type() != var2.Type()) {
    return false;
  }

  framework::Tensor t1, t2;

  if (var1.IsType<framework::LoDTensor>()) {
    framework::TensorCopySync(var1.Get<framework::LoDTensor>(),
                              platform::CPUPlace(), &t1);
    framework::TensorCopySync(var2.Get<framework::LoDTensor>(),
                              platform::CPUPlace(), &t2);
  } else {
    auto& s1 = var1.Get<framework::SelectedRows>();
    auto& s2 = var2.Get<framework::SelectedRows>();

    if (s1.height() != s2.height()) {
      return false;
    }

    if (s1.rows().size() != s2.rows().size()) {
      return false;
    }

    auto row1_data = s1.rows().data();
    auto row2_data = s2.rows().data();
    if (std::memcmp(row1_data, row2_data,
                    s1.rows().size() * sizeof(*row1_data)) != 0) {
      return false;
    }

    framework::TensorCopySync(var1.Get<framework::SelectedRows>().value(),
                              platform::CPUPlace(), &t1);
    framework::TensorCopySync(var2.Get<framework::SelectedRows>().value(),
                              platform::CPUPlace(), &t2);
  }

  if (t1.type() != t2.type() || t1.dims() != t2.dims()) {
    return false;
  }

  auto* t1_p = t1.data<void>();
  auto* t2_p = t2.data<void>();
  return std::memcmp(t1_p, t2_p,
                     t1.numel() * framework::SizeOfType(t1.type())) == 0;
}

template <typename T>
static framework::Variable RandomTensor(const framework::DDim& dims,
                                        const platform::Place& place,
                                        int low = -10, int high = 10) {
  framework::Tensor cpu_tensor;
  cpu_tensor.Resize(dims);
  auto* ptr = cpu_tensor.mutable_data<T>(platform::CPUPlace());
  std::uniform_int_distribution<int> dist(low, high);
  std::random_device rd;
  std::mt19937 engine(rd());
  for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
    ptr[i] = dist(engine);
  }

  framework::Variable ret;
  framework::TensorCopySync(cpu_tensor, place,
                            ret.GetMutable<framework::LoDTensor>());
  return ret;
}

template <typename T>
static framework::Variable RandomSelectedRows(framework::DDim dims,
                                              const platform::Place& place,
                                              int64_t row_number, int low = -10,
                                              int high = 10) {
  auto height = dims[0];
  dims[0] = row_number;

  framework::Variable ret;
  auto* sr = ret.GetMutable<framework::SelectedRows>();
  auto tensor_var = RandomTensor<T>(dims, place, low, high);
  sr->mutable_value()->ShareDataWith(
      tensor_var.template Get<framework::LoDTensor>());
  sr->set_height(height);
  sr->mutable_rows()->resize(row_number);
  auto* row_data = sr->mutable_rows()->data();
  std::uniform_int_distribution<int64_t> dist(0, height - 1);
  std::random_device rd;
  std::mt19937 engine(rd());
  for (int64_t i = 0; i < dims[0]; ++i) {
    row_data[i] = dist(engine);
  }
  return ret;
}

static std::unique_ptr<GradientAccumulator> CreateAccumulator(
    const std::shared_ptr<VariableWrapper>& var, bool sort_gradient) {
  if (sort_gradient) {
    return std::unique_ptr<GradientAccumulator>(
        new SortedGradientAccumulator(var.get()));
  } else {
    return std::unique_ptr<GradientAccumulator>(
        new EagerGradientAccumulator(var.get()));
  }
}

static void TestGradientAccumulatorTestUnchangeInput(
    const platform::Place& place, bool sort_gradient) {
  framework::DDim dim{10, 20};
  int64_t maximum_row_number = 100;

  std::uniform_int_distribution<int64_t> dist(1, maximum_row_number);
  int seed;
  {
    std::random_device rd;
    seed = rd();
  }

  std::mt19937 engine(seed);

  auto create_var = [&](bool use_tensor) {
    if (use_tensor) {
      return RandomTensor<float>(dim, place);
    } else {
      return RandomSelectedRows<float>(dim, place, dist(engine));
    }
  };

  std::vector<bool> use_tensors = {false, true};

  for (auto use_tensor1 : use_tensors) {
    for (auto use_tensor2 : use_tensors) {
      auto g_var1 = std::make_shared<VariableWrapper>("g_var1");
      g_var1->SetOverridedStopGradient(false);
      auto g_accum1 = CreateAccumulator(g_var1, sort_gradient);
      g_accum1->IncreaseRefCnt();
      g_accum1->IncreaseRefCnt();

      auto g_var2 = std::make_shared<VariableWrapper>("g_var2");
      g_var2->SetOverridedStopGradient(false);
      auto g_accum2 = CreateAccumulator(g_var2, sort_gradient);
      g_accum2->IncreaseRefCnt();
      g_accum2->IncreaseRefCnt();

      auto var1 = create_var(use_tensor1);
      auto var_wrapper1_1 = std::make_shared<VariableWrapper>("tmp1_1");
      auto var_wrapper2_1 = std::make_shared<VariableWrapper>("tmp2_1");
      CopyVar(var1, var_wrapper1_1->MutableVar());
      CopyVar(var1, var_wrapper2_1->MutableVar());

      auto var2 = create_var(use_tensor2);
      auto var_wrapper1_2 = std::make_shared<VariableWrapper>("tmp1_2");
      auto var_wrapper2_2 = std::make_shared<VariableWrapper>("tmp2_2");
      CopyVar(var2, var_wrapper1_2->MutableVar());
      CopyVar(var2, var_wrapper2_2->MutableVar());

      g_accum1->Add(var_wrapper1_1, 0, false);
      g_accum1->Add(var_wrapper1_2, 1, false);

      g_accum2->Add(var_wrapper2_1, 0, true);
      g_accum2->Add(var_wrapper2_2, 1, true);

      ASSERT_TRUE(IsEqualVar(var_wrapper2_1->Var(), var1));
      ASSERT_TRUE(IsEqualVar(var_wrapper2_2->Var(), var2));
      ASSERT_TRUE(IsEqualVar(g_var1->Var(), g_var2->Var()));
    }
  }
}

TEST(test_gradient_accumulator, test_unchange_input) {
  for (auto sort_gradient : {false, true}) {
    TestGradientAccumulatorTestUnchangeInput(platform::CPUPlace(),
                                             sort_gradient);
#ifdef PADDLE_WITH_CUDA
    TestGradientAccumulatorTestUnchangeInput(platform::CUDAPlace(0),
                                             sort_gradient);
#endif
  }
}

}  // namespace imperative
}  // namespace paddle
