// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/kernel/tensor_kernels.h"

#include <iostream>
#include <vector>

#include "paddle/infrt/common/global.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/kernel_utils.h"
#include "paddle/infrt/tensor/dense_host_tensor.h"
#include "paddle/infrt/tensor/dense_tensor_view.h"
#include "paddle/infrt/tensor/tensor_map.h"
#include "paddle/infrt/tensor/tensor_shape.h"

#ifdef INFRT_WITH_PHI
#include "paddle/phi/core/dense_tensor.h"
#endif

namespace infrt {
namespace kernel {
using namespace host_context;  // NOLINT
using namespace tensor;        // NOLINT

/// ===== Kernel begin ====

template <typename T>
DenseHostTensor CreateUninitTensor(Attribute<std::vector<int64_t>> shape) {
  const auto &shape_data = shape.get();
  auto array = llvm::ArrayRef<int64_t>(shape_data.data(), shape_data.size());
  auto type = GetDType<T>();
  return DenseHostTensor(TensorShape(array), type);
}

void PrintTensor(const DenseHostTensor &tensor) {
  std::cout << tensor << std::endl;
}

template <typename T>
void FillTensorWithConstant(Attribute<T> v, DenseHostTensor *tensor) {
  MutableDTArrayView<T>(tensor).Fill(v.get());
}

TensorMap LoadParams(Attribute<std::string> path) {
  return *(infrt::tensor::LoadParams(path.get()));
}

DenseHostTensor TensorMapGetTensor(TensorMap map, Attribute<std::string> name) {
  auto it = map.find(name.get());
  CHECK(it != map.end()) << "No tensor called " << name.get()
                         << " in the TensorMap";
  return *it->second;
}

int32_t TensorMapGetSize(TensorMap map) { return map.size(); }

// TODO(wilber): Maybe we should place TensorList type in dt dialect.
#ifdef INFRT_WITH_PHI
::Tensor TensorListGetTensor(std::vector<::Tensor *> list,
                             Attribute<int32_t> idx) {
  CHECK_LT(idx.get(), static_cast<int>(list.size()))
      << "idx should less than list size";
  return *list[idx.get()];
}

int32_t TensorListGetSize(const std::vector<::Tensor *> &list) {
  return list.size();
}
#endif

DenseHostTensor ShallowCopyTensor(DenseHostTensor v) { return v; }

template <typename T>
void NaiveElementwiseAdd(const DenseHostTensor &x,
                         const DenseHostTensor &y,
                         DenseHostTensor *out) {
  CHECK_EQ(x.shape().GetNumElements(), y.shape().GetNumElements());

  // Infer shape
  *out = DenseHostTensor(x.shape(), GetDType<T>());

  const T *x_data = static_cast<T *>(x.raw_data());
  const T *y_data = static_cast<T *>(y.raw_data());
  T *out_data = static_cast<T *>(out->raw_data());
  for (size_t i = 0, n = x.shape().GetNumElements(); i < n; i++) {
    out_data[i] = x_data[i] + y_data[i];
  }
}

//! A naive implementation for x matmul w
template <typename T>
void NaiveMatmul(const DenseHostTensor &x,
                 const DenseHostTensor &w,
                 DenseHostTensor *out) {
  CHECK_EQ(x.shape().GetRank(), 2);
  CHECK_EQ(w.shape().GetRank(), 2);
  CHECK_EQ(x.shape().GetDim(x.shape().GetRank() - 1), w.shape().GetDim(0));
  std::vector<int64_t> out_dims({x.shape().GetDim(0), w.shape().GetDim(1)});
  *out = DenseHostTensor(TensorShape(out_dims), GetDType<T>());

  auto *out_data = static_cast<T *>(out->raw_data());
  auto *x_data = static_cast<const T *>(x.raw_data());
  auto *w_data = static_cast<const T *>(w.raw_data());

  const int M = x.shape().GetDim(0);
  const int K = x.shape().GetDim(1);
  const int N = w.shape().GetDim(1);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      out_data[i * N + j] = 0;
      for (int k = 0; k < K; k++) {
        out_data[i * N + j] += x_data[i * K + k] * w_data[k * N + j];
      }
    }
  }
}

/// ===== Kernel end ====

void RegisterTensorKernels(host_context::KernelRegistry *registry) {
  registry->AddKernel("dt.create_uninit_tensor.f32",
                      INFRT_KERNEL(CreateUninitTensor<float>),
                      {"shape"});
  registry->AddKernel("dt.print_tensor", INFRT_KERNEL(PrintTensor));
  registry->AddKernel("dt.fill_tensor_with_constant.f32",
                      INFRT_KERNEL(FillTensorWithConstant<float>),
                      {"value"});
  registry->AddKernel("dt.fill_tensor_with_constant.f64",
                      INFRT_KERNEL(FillTensorWithConstant<double>),
                      {"value"});

  // TensorMap related methods.
  registry->AddKernel("dt.load_params", INFRT_KERNEL(LoadParams));
  registry->AddKernel("dt.tensor_map_get_tensor",
                      INFRT_KERNEL(TensorMapGetTensor));
  registry->AddKernel("dt.tensor_map_get_size", INFRT_KERNEL(TensorMapGetSize));

// TensorList related methods.
#ifdef INFRT_WITH_PHI
  registry->AddKernel(
      "dt.tensor_list_get_tensor", INFRT_KERNEL(TensorListGetTensor), {"id"});
  registry->AddKernel("dt.tensor_list_get_size",
                      INFRT_KERNEL(TensorListGetSize));
#endif

  registry->AddKernel("dt.shallow_copy_tensor",
                      INFRT_KERNEL(ShallowCopyTensor));

  // Naive kernels.
  registry->AddKernel("dt.naive_elementwise_add.f32",
                      INFRT_KERNEL(NaiveElementwiseAdd<float>));
  registry->AddKernel("dt.naive_matmul.f32", INFRT_KERNEL(NaiveMatmul<float>));
}

}  // namespace kernel
}  // namespace infrt
