// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/repeat_tensor2index_tensor.h"

namespace phi {
namespace funcs {

template <typename Context, typename RepeatsT>
void RepeatsTensor2IndexTensorFunctor<Context, RepeatsT>::operator()(
    const Context &dev_ctx, const DenseTensor &repeats, DenseTensor *index) {
  DenseTensor repeats_cpu_copy;
  if (repeats.place().GetType() != phi::AllocationType::CPU) {
    phi::Copy(dev_ctx, repeats, phi::CPUPlace(), true, &repeats_cpu_copy);
  }
  const RepeatsT *repeats_data =
      repeats.place().GetType() == phi::AllocationType::CPU
          ? repeats.data<RepeatsT>()
          : repeats_cpu_copy.data<RepeatsT>();

  int64_t index_size = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    PADDLE_ENFORCE_GE(repeats_data[i],
                      0,
                      common::errors::InvalidArgument(
                          "repeats must grater or equal than 0, but got %d",
                          repeats_data[i]));
    index_size += repeats_data[i];
  }
  std::vector<RepeatsT> index_vec(index_size);
  int offset = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    std::fill_n(index_vec.begin() + offset, repeats_data[i], i);
    offset += repeats_data[i];
  }
  index->Resize(common::make_ddim({index_size}));

  phi::TensorFromVector<RepeatsT>(index_vec, dev_ctx, index);
}

template <typename RepeatsT>
void RepeatsTensor2IndexTensorFunctor<phi::CPUContext, RepeatsT>::operator()(
    const phi::CPUContext &dev_ctx,
    const DenseTensor &repeats,
    DenseTensor *index) {
  const RepeatsT *repeats_data = repeats.data<RepeatsT>();

  int64_t index_size = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    PADDLE_ENFORCE_GE(repeats_data[i],
                      0,
                      common::errors::InvalidArgument(
                          "repeats must grater or equal than 0, but got %d",
                          repeats_data[i]));
    index_size += repeats_data[i];
  }
  std::vector<RepeatsT> index_vec(index_size);
  int offset = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    std::fill_n(index_vec.begin() + offset, repeats_data[i], i);
    offset += repeats_data[i];
  }
  index->Resize(common::make_ddim({index_size}));

  phi::TensorFromVector<RepeatsT>(index_vec, dev_ctx, index);
}

template class RepeatsTensor2IndexTensorFunctor<phi::CPUContext, int>;
template class RepeatsTensor2IndexTensorFunctor<phi::CPUContext, int64_t>;

#ifdef PADDLE_WITH_XPU
template <typename RepeatsT>
void RepeatsTensor2IndexTensorFunctor<phi::XPUContext, RepeatsT>::operator()(
    const phi::XPUContext &dev_ctx,
    const DenseTensor &repeats,
    DenseTensor *index) {
  DenseTensor repeats_cpu_copy;
  phi::Copy(dev_ctx, repeats, phi::CPUPlace(), true, &repeats_cpu_copy);
  const RepeatsT *repeats_data = repeats_cpu_copy.data<RepeatsT>();

  int64_t index_size = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    PADDLE_ENFORCE_GE(repeats_data[i],
                      0,
                      common::errors::InvalidArgument(
                          "repeats must grater or equal than 0, but got %d",
                          repeats_data[i]));
    index_size += repeats_data[i];
  }
  std::vector<RepeatsT> index_vec(index_size);
  int offset = 0;
  for (int i = 0; i < repeats.dims()[0]; i++) {
    std::fill_n(index_vec.begin() + offset, repeats_data[i], i);
    offset += repeats_data[i];
  }
  index->Resize(common::make_ddim({index_size}));

  phi::TensorFromVector<RepeatsT>(index_vec, dev_ctx, index);
}

template class RepeatsTensor2IndexTensorFunctor<phi::XPUContext, int>;
template class RepeatsTensor2IndexTensorFunctor<phi::XPUContext, int64_t>;
#endif

}  // namespace funcs
}  // namespace phi
