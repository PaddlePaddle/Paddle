// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/api/backward/backward_api.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void CSplitKernel(const Context& dev_ctx,
                  const DenseTensor& x_in,
                  int rank,
                  int nranks,
                  int ring_id UNUSED,
                  bool use_calc_stream UNUSED,
                  bool use_model_parallel UNUSED,
                  DenseTensor* out) {
  auto x = &x_in;
  PADDLE_ENFORCE_GE(rank,
                    0,
                    common::errors::PreconditionNotMet(
                        "The value of rank (%d) for c_split must be "
                        "greater than or equal to 0.",
                        rank));
  PADDLE_ENFORCE_GE(nranks,
                    2,
                    common::errors::PreconditionNotMet(
                        "The value of nranks (%d) for c_split must be "
                        "greater than or equal to 2.",
                        nranks));
  PADDLE_ENFORCE_LT(rank,
                    nranks,
                    common::errors::PreconditionNotMet(
                        "The value of rank (%d) for c_split must be "
                        "less than that of nranks (%d).",
                        rank,
                        nranks));

  auto dims = x->dims();
  auto dims_size = dims.size();

  dims[dims_size - 1] /= nranks;
  dev_ctx.template Alloc<T>(out);
  out->Resize(dims);
  std::vector<int64_t> split_list(nranks, dims[dims_size - 1]);
  int axis = dims_size - 1;

  auto x_tmp = std::make_shared<phi::DenseTensor>();
  x_tmp->ShareDataWith(*x);
  paddle::Tensor x_tensor(x_tmp);
  auto outputs = paddle::experimental::split(x_tensor, split_list, axis);
  out->ShareDataWith(
      *reinterpret_cast<phi::DenseTensor*>(outputs[rank].impl().get()));
}
}  // namespace phi

PD_REGISTER_KERNEL(c_split,
                   Custom,
                   ALL_LAYOUT,
                   phi::CSplitKernel,
                   float,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
