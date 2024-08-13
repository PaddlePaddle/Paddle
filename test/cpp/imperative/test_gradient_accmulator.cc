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
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace imperative {

TEST(Test__SelectedRowsMerge_Test, SelectedRowsMerge) {
  phi::CPUPlace cpu;

  std::vector<int64_t> rows{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int64_t table_size = 10;
  int64_t embedding_width = 10;

  auto sr1 = std::make_shared<phi::SelectedRows>(rows, table_size);
  auto sr2 = std::make_shared<phi::SelectedRows>(rows, table_size);

  // initialize a sparse table 1
  sr1->mutable_value()->Resize(
      common::make_ddim({table_size, embedding_width}));
  auto* data_sr1 = sr1->mutable_value()->mutable_data<float>(cpu);
  for (int64_t i = 0; i < table_size; ++i) {
    for (int64_t j = 0; j < embedding_width; ++j) {
      data_sr1[i * embedding_width + j] = static_cast<float>(i);
    }
  }

  // initialize a sparse table 2
  sr2->mutable_value()->Resize(
      common::make_ddim({table_size, embedding_width}));
  auto* data_sr2 = sr2->mutable_value()->mutable_data<float>(cpu);
  for (int64_t i = 0; i < table_size; ++i) {
    for (int64_t j = 0; j < embedding_width; ++j) {
      data_sr2[i * embedding_width + j] = static_cast<float>(i);
    }
  }
  // new 2 phi::Tensor
  paddle::Tensor t1(sr1);
  paddle::Tensor t2(sr2);

  // call SelectedRowsMerge
  auto new_buffer =
      paddle::imperative::SelectedRowsMerge<paddle::Tensor>(t1, t2);
  auto* new_buffer_tensor =
      static_cast<phi::SelectedRows*>(new_buffer->impl().get());
  auto* new_buffer_data_sr1 =
      new_buffer_tensor->mutable_value()->mutable_data<float>(cpu);

  // verify the MergeAdd result
  for (int64_t i = 0; i < table_size; ++i) {
    for (int64_t j = 0; j < embedding_width; ++j) {
      EXPECT_EQ(new_buffer_data_sr1[i * embedding_width + j],
                (static_cast<float>(i) + static_cast<float>(i)));
    }
  }
}

template <typename Place1, typename Place2, typename T>
int TensorAddTest(Place1 place1, Place2 place2, T t1, T t2) {
  framework::Variable var1;
  framework::Variable var2;
  std::vector<T> src_data(10, t1);
  std::vector<T> dst_data(10, t2);
  std::vector<T> result;
  phi::CPUPlace src_place;
  for (unsigned int i = 0; i < 10; i++) {
    result.emplace_back(src_data[i] + dst_data[i]);
  }

  std::vector<int64_t> dims = {2, 5};
  auto* src = var1.GetMutable<phi::DenseTensor>();
  auto* dst = var2.GetMutable<phi::DenseTensor>();
  src->Resize(common::make_ddim(dims));
  dst->Resize(common::make_ddim(dims));
  auto* src_mutable = src->mutable_data<T>(place1);
  auto* dst_mutable = dst->mutable_data<T>(place2);

  if (!std::is_same<Place1, phi::GPUPlace>::value) {
    paddle::memory::Copy(place1,
                         src_mutable,
                         src_place,
                         src_data.data(),
                         sizeof(T) * src_data.size());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else {
    paddle::memory::Copy(place1,
                         src_mutable,
                         src_place,
                         src_data.data(),
                         sizeof(T) * src_data.size(),
                         0);
#endif
  }

  if (!std::is_same<Place2, phi::GPUPlace>::value) {
    paddle::memory::Copy(place2,
                         dst_mutable,
                         src_place,
                         dst_data.data(),
                         sizeof(T) * dst_data.size());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else {
    paddle::memory::Copy(place2,
                         dst_mutable,
                         src_place,
                         dst_data.data(),
                         sizeof(T) * dst_data.size(),
                         0);
#endif
  }
  imperative::TensorAdd<framework::Variable>(var1, &var2);
  phi::DenseTensor rlt;
  phi::CPUPlace rlt_place;
  framework::TensorCopySync(*dst, rlt_place, &rlt);

  for (unsigned int i = 0; i < rlt.numel(); i++) {
    if (rlt.data<T>()[i] != result[i]) return 1;
  }

  return 0;
}

TEST(test_add_functor, add_functor) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  phi::GPUPlace gpu_place(0);
#endif
  phi::CPUPlace cpu_place;

  int cpu_res = 1;

  // float32
  cpu_res = TensorAddTest(
      cpu_place, cpu_place, static_cast<float>(1.0), static_cast<float>(2.0));
  EXPECT_EQ(cpu_res, 0);
  // float16
  cpu_res = TensorAddTest(cpu_place,
                          cpu_place,
                          static_cast<phi::dtype::float16>(1.0),
                          static_cast<phi::dtype::float16>(2.0));
  EXPECT_EQ(cpu_res, 0);
  // double
  cpu_res = TensorAddTest(
      cpu_place, cpu_place, static_cast<double>(1.0), static_cast<double>(2.0));
  EXPECT_EQ(cpu_res, 0);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  int gpu_res = 1;
  gpu_res = TensorAddTest(gpu_place, gpu_place, 1.0, 0.0);
  EXPECT_EQ(gpu_res, 0);
  gpu_res = TensorAddTest(
      gpu_place, gpu_place, static_cast<double>(1.0), static_cast<double>(2.0));
  EXPECT_EQ(gpu_res, 0);

  // normal
  gpu_res = TensorAddTest(
      gpu_place, gpu_place, static_cast<float>(1.0), static_cast<float>(2.0));
  EXPECT_EQ(gpu_res, 0);
  gpu_res = TensorAddTest(gpu_place,
                          gpu_place,
                          static_cast<phi::dtype::float16>(1.0),
                          static_cast<phi::dtype::float16>(2.0));
  EXPECT_EQ(gpu_res, 0);
  // different places
  gpu_res = TensorAddTest(
      cpu_place, gpu_place, static_cast<float>(1.0), static_cast<float>(2.0));
  EXPECT_EQ(gpu_res, 0);
  gpu_res = TensorAddTest(
      gpu_place, cpu_place, static_cast<float>(1.0), static_cast<float>(2.0));
  EXPECT_EQ(gpu_res, 0);
  gpu_res = TensorAddTest(cpu_place,
                          gpu_place,
                          static_cast<phi::dtype::float16>(1.0),
                          static_cast<phi::dtype::float16>(2.0));
  EXPECT_EQ(gpu_res, 0);
  gpu_res = TensorAddTest(gpu_place,
                          cpu_place,
                          static_cast<phi::dtype::float16>(1.0),
                          static_cast<phi::dtype::float16>(2.0));
  EXPECT_EQ(gpu_res, 0);
#endif

#ifdef PADDLE_WITH_XPU
  phi::XPUPlace xpu_place(0);
  int xpu_res = 1;
  // normal
  xpu_res = TensorAddTest(
      xpu_place, xpu_place, static_cast<float>(1.0), static_cast<float>(2.0));
  EXPECT_EQ(xpu_res, 0);
  xpu_res = TensorAddTest(xpu_place,
                          xpu_place,
                          static_cast<phi::dtype::float16>(1.0),
                          static_cast<phi::dtype::float16>(2.0));
  EXPECT_EQ(xpu_res, 0);
  xpu_res = TensorAddTest(
      xpu_place, xpu_place, static_cast<double>(1.0), static_cast<double>(2.0));
  EXPECT_EQ(xpu_res, 0);
  // different places
  xpu_res = TensorAddTest(
      cpu_place, xpu_place, static_cast<float>(1.0), static_cast<float>(2.0));
  EXPECT_EQ(xpu_res, 0);
  xpu_res = TensorAddTest(
      xpu_place, cpu_place, static_cast<float>(1.0), static_cast<float>(2.0));
  EXPECT_EQ(xpu_res, 0);
  xpu_res = TensorAddTest(cpu_place,
                          xpu_place,
                          static_cast<phi::dtype::float16>(1.0),
                          static_cast<phi::dtype::float16>(2.0));
  EXPECT_EQ(xpu_res, 0);
  xpu_res = TensorAddTest(xpu_place,
                          cpu_place,
                          static_cast<phi::dtype::float16>(1.0),
                          static_cast<phi::dtype::float16>(2.0));
  EXPECT_EQ(xpu_res, 0);
  xpu_res = TensorAddTest(
      cpu_place, xpu_place, static_cast<double>(1.0), static_cast<double>(2.0));
  EXPECT_EQ(xpu_res, 0);
  xpu_res = TensorAddTest(
      xpu_place, cpu_place, static_cast<double>(1.0), static_cast<double>(2.0));
  EXPECT_EQ(xpu_res, 0);
#endif
}

TEST(test_add_functor, exception) {
  phi::GPUPinnedPlace cuda_pinned_place;
  phi::GPUPlace cuda_place(0);
  phi::CPUPlace cpu_place;

  ASSERT_ANY_THROW(TensorAddTest(cpu_place, cpu_place, 1, 0));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  ASSERT_ANY_THROW(
      TensorAddTest(cuda_pinned_place, cuda_pinned_place, 1.0, 0.0));
  ASSERT_ANY_THROW(TensorAddTest(cuda_pinned_place,
                                 cuda_pinned_place,
                                 static_cast<phi::dtype::float16>(1.0),
                                 static_cast<phi::dtype::float16>(2.0)));
#endif
}

static void CopyVar(const framework::Variable& var,
                    framework::Variable* dst_ptr) {
  auto& dst = *dst_ptr;
  dst.Clear();
  if (var.IsType<phi::DenseTensor>()) {
    const auto& src_tensor = var.Get<phi::DenseTensor>();
    auto* dst_tensor = dst.GetMutable<phi::DenseTensor>();
    framework::TensorCopySync(src_tensor, src_tensor.place(), dst_tensor);
  } else {
    const auto& src_selected_rows = var.Get<phi::SelectedRows>();
    auto* dst_selected_rows = dst.GetMutable<phi::SelectedRows>();
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

  phi::DenseTensor t1, t2;

  if (var1.IsType<phi::DenseTensor>()) {
    framework::TensorCopySync(
        var1.Get<phi::DenseTensor>(), phi::CPUPlace(), &t1);
    framework::TensorCopySync(
        var2.Get<phi::DenseTensor>(), phi::CPUPlace(), &t2);
  } else {
    auto& s1 = var1.Get<phi::SelectedRows>();
    auto& s2 = var2.Get<phi::SelectedRows>();

    if (s1.height() != s2.height()) {
      return false;
    }

    if (s1.rows().size() != s2.rows().size()) {
      return false;
    }

    auto row1_data = s1.rows().data();
    auto row2_data = s2.rows().data();
    if (std::memcmp(
            row1_data, row2_data, s1.rows().size() * sizeof(*row1_data)) != 0) {
      return false;
    }

    framework::TensorCopySync(
        var1.Get<phi::SelectedRows>().value(), phi::CPUPlace(), &t1);
    framework::TensorCopySync(
        var2.Get<phi::SelectedRows>().value(), phi::CPUPlace(), &t2);
  }

  if (t1.type() != t2.type() || t1.dims() != t2.dims()) {
    return false;
  }

  auto* t1_p = t1.data();
  auto* t2_p = t2.data();
  return std::memcmp(
             t1_p,
             t2_p,
             t1.numel() * framework::SizeOfType(
                              framework::TransToProtoVarType(t1.dtype()))) == 0;
}

template <typename T>
static framework::Variable RandomTensor(const phi::DDim& dims,
                                        const phi::Place& place,
                                        int low = -10,
                                        int high = 10) {
  phi::DenseTensor cpu_tensor;
  cpu_tensor.Resize(dims);
  auto* ptr = cpu_tensor.mutable_data<T>(phi::CPUPlace());
  std::uniform_int_distribution<int> dist(low, high);
  std::random_device rd;
  std::mt19937 engine(rd());
  for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
    ptr[i] = dist(engine);
  }

  framework::Variable ret;
  framework::TensorCopySync(
      cpu_tensor, place, ret.GetMutable<phi::DenseTensor>());
  return ret;
}

template <typename T>
static framework::Variable RandomSelectedRows(phi::DDim dims,
                                              const phi::Place& place,
                                              int64_t row_number,
                                              int low = -10,
                                              int high = 10) {
  auto height = dims[0];
  dims[0] = row_number;

  framework::Variable ret;
  auto* sr = ret.GetMutable<phi::SelectedRows>();
  auto tensor_var = RandomTensor<T>(dims, place, low, high);
  sr->mutable_value()->ShareDataWith(
      tensor_var.template Get<phi::DenseTensor>());
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
  if (sort_gradient) {  // NOLINT
    return std::unique_ptr<GradientAccumulator>(
        new SortedGradientAccumulator(var.get()));
  } else {
    return std::unique_ptr<GradientAccumulator>(
        new EagerGradientAccumulator(var.get()));
  }
}

static void TestGradientAccumulatorTestUnchangeInput(const phi::Place& place,
                                                     bool sort_gradient) {
  phi::DDim dim{10, 20};
  int64_t maximum_row_number = 100;

  std::uniform_int_distribution<int64_t> dist(1, maximum_row_number);
  int seed = 0;
  {
    std::random_device rd;
    seed = static_cast<int>(rd());
  }

  std::mt19937 engine(seed);

  auto create_var = [&](bool use_tensor) {
    if (use_tensor) {  // NOLINT
      return RandomTensor<float>(dim, place);
    } else {
      return RandomSelectedRows<float>(dim, place, dist(engine));
    }
  };

  std::vector<bool> use_tensors = {false, true};

  for (auto use_tensor1 : use_tensors) {
    for (auto use_tensor2 : use_tensors) {
      /** g_accum1 && g_accum2: has not been initialized
       *    test accumulate on this graph
       */
      auto g_var1 = std::make_shared<VariableWrapper>("g_var1");
      g_var1->SetOverriddenStopGradient(false);
      auto g_accum1 = CreateAccumulator(g_var1, sort_gradient);
      g_accum1->IncreaseRefCnt();
      g_accum1->IncreaseRefCnt();

      auto g_var2 = std::make_shared<VariableWrapper>("g_var2");
      g_var2->SetOverriddenStopGradient(false);
      auto g_accum2 = CreateAccumulator(g_var2, sort_gradient);
      g_accum2->IncreaseRefCnt();
      g_accum2->IncreaseRefCnt();

      auto var1 = create_var(use_tensor1);
      auto var_wrapper1_1 = std::make_shared<VariableWrapper>("tmp1_1");
      auto var_wrapper2_1 = std::make_shared<VariableWrapper>("tmp2_1");

      ASSERT_EQ(var_wrapper1_1->IsEmpty(), true);
      CopyVar(var1, var_wrapper1_1->MutableVar());
      ASSERT_EQ(var_wrapper1_1->IsEmpty(), false);

      ASSERT_EQ(var_wrapper2_1->IsEmpty(), true);
      CopyVar(var1, var_wrapper2_1->MutableVar());
      ASSERT_EQ(var_wrapper2_1->IsEmpty(), false);

      auto var2 = create_var(use_tensor2);
      auto var_wrapper1_2 = std::make_shared<VariableWrapper>("tmp1_2");
      auto var_wrapper2_2 = std::make_shared<VariableWrapper>("tmp2_2");
      CopyVar(var2, var_wrapper1_2->MutableVar());
      CopyVar(var2, var_wrapper2_2->MutableVar());

      // g_accum1: inner_var_ = var1 + var2
      g_accum1->SumGrad(var_wrapper1_1, 0, false);
      g_accum1->SumGrad(var_wrapper1_2, 1, false);
      ASSERT_EQ(g_accum1->CurCnt(), g_accum1->RefCnt());
      ASSERT_TRUE(g_accum1->SumGradCompleted());
      // g_accum1: inner_var_ -> var_
      g_accum1->AccumulateGrad();

      // g_accum2: inner_var_ = var1 + var2
      g_accum2->SumGrad(var_wrapper2_1, 0, true);
      g_accum2->SumGrad(var_wrapper2_2, 1, true);
      ASSERT_EQ(g_accum2->CurCnt(), g_accum2->RefCnt());
      ASSERT_TRUE(g_accum2->SumGradCompleted());
      // g_accum2: inner_var_ -> var_
      g_accum2->AccumulateGrad();

      ASSERT_TRUE(IsEqualVar(var_wrapper2_1->Var(), var1));
      ASSERT_TRUE(IsEqualVar(var_wrapper2_2->Var(), var2));
      ASSERT_TRUE(IsEqualVar(g_var1->Var(), g_var2->Var()));

      /** g_accum3 && g_accum4: has been initialized
       *    test accumulate on previous graph
       */
      auto var3 = create_var(use_tensor1);
      auto var_wrapper3_3 = std::make_shared<VariableWrapper>("tmp1_3");
      auto var_wrapper4_3 = std::make_shared<VariableWrapper>("tmp2_3");
      var_wrapper3_3->SetOverriddenStopGradient(false);
      var_wrapper4_3->SetOverriddenStopGradient(false);
      CopyVar(var3, var_wrapper3_3->MutableVar());
      CopyVar(var3, var_wrapper4_3->MutableVar());

      auto g_accum3 = CreateAccumulator(var_wrapper3_3, sort_gradient);
      g_accum3->IncreaseRefCnt();
      auto g_accum4 = CreateAccumulator(var_wrapper4_3, sort_gradient);
      g_accum4->IncreaseRefCnt();

      auto var4 = create_var(use_tensor2);
      auto var_wrapper3_4 = std::make_shared<VariableWrapper>("tmp1_4");
      auto var_wrapper4_4 = std::make_shared<VariableWrapper>("tmp2_4");
      CopyVar(var4, var_wrapper3_4->MutableVar());
      CopyVar(var4, var_wrapper4_4->MutableVar());

      g_accum3->SumGrad(var_wrapper3_4, 0, false);
      ASSERT_TRUE(g_accum3->SumGradCompleted());
      // g_accum4: var_(var_wrapper3_3) + inner_var_ -> var_
      g_accum3->AccumulateGrad();

      g_accum4->SumGrad(var_wrapper4_4, 0, false);
      ASSERT_TRUE(g_accum4->SumGradCompleted());
      // g_accum4: var_(var_wrapper4_3) + inner_var_ -> var_
      g_accum4->AccumulateGrad();

      ASSERT_TRUE(IsEqualVar(var_wrapper3_3->Var(), var_wrapper4_3->Var()));
    }
  }
}

TEST(test_gradient_accumulator, test_unchange_input) {
  for (auto sort_gradient : {false, true}) {
    TestGradientAccumulatorTestUnchangeInput(phi::CPUPlace(), sort_gradient);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    TestGradientAccumulatorTestUnchangeInput(phi::GPUPlace(0), sort_gradient);
#endif
  }
}

}  // namespace imperative
}  // namespace paddle
