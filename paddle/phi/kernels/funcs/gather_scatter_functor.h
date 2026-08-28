/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"

#pragma once

namespace phi {
namespace funcs {

#define Instantiate_Template_Function(func)                                  \
  Instantiate_Template_Function_index_t(                                     \
      func, int) Instantiate_Template_Function_index_t(func, float)          \
      Instantiate_Template_Function_index_t(                                 \
          func, double) Instantiate_Template_Function_index_t(func, int64_t) \
          Instantiate_Template_Function_index_t(func, phi::float16)          \
              Instantiate_Template_Function_index_t(func, phi::bfloat16)     \
                  Instantiate_Template_Function_index_t(func, unsigned char) \
                      Instantiate_Template_Function_index_t(func, int16_t)

#define Instantiate_Template_Function_index_t(func, tensor_t)      \
  template void func<tensor_t, int>(DenseTensor input,             \
                                    int dim,                       \
                                    const DenseTensor& index,      \
                                    DenseTensor result,            \
                                    bool include_self,             \
                                    const DeviceContext& dev_ctx); \
  template void func<tensor_t, int64_t>(DenseTensor input,         \
                                        int dim,                   \
                                        const DenseTensor& index,  \
                                        DenseTensor result,        \
                                        bool include_self,         \
                                        const DeviceContext& dev_ctx);

#define Instantiate_Template_Function_With_Out(func)                           \
  Instantiate_Template_Function_index_t_With_Out(func, int)                    \
      Instantiate_Template_Function_index_t_With_Out(func, float)              \
          Instantiate_Template_Function_index_t_With_Out(func, double)         \
              Instantiate_Template_Function_index_t_With_Out(func, int64_t)    \
                  Instantiate_Template_Function_index_t_With_Out(func,         \
                                                                 phi::float16) \
                      Instantiate_Template_Function_index_t_With_Out(          \
                          func, phi::bfloat16)                                 \
                          Instantiate_Template_Function_index_t_With_Out(      \
                              func, unsigned char)                             \
                              Instantiate_Template_Function_index_t_With_Out(  \
                                  func, int16_t)
#define Instantiate_Template_Function_index_t_With_Out(func, tensor_t) \
  template void func<tensor_t, int>(DenseTensor input,                 \
                                    int dim,                           \
                                    const DenseTensor& index,          \
                                    const DenseTensor& out,            \
                                    const DenseTensor& self,           \
                                    const DenseTensor& value,          \
                                    DenseTensor result,                \
                                    const std::string& reduce,         \
                                    bool include_self,                 \
                                    const DeviceContext& dev_ctx);     \
  template void func<tensor_t, int64_t>(DenseTensor input,             \
                                        int dim,                       \
                                        const DenseTensor& index,      \
                                        const DenseTensor& out,        \
                                        const DenseTensor& self,       \
                                        const DenseTensor& value,      \
                                        DenseTensor result,            \
                                        const std::string& reduce,     \
                                        bool include_self,             \
                                        const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_gather_kernel(DenseTensor self,
                       int dim,
                       const DenseTensor& index,
                       DenseTensor result,
                       bool include_self,
                       const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_assign_kernel(DenseTensor self,
                               int dim,
                               const DenseTensor& index,
                               DenseTensor src,
                               bool include_self,
                               const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_add_kernel(DenseTensor self,
                            int dim,
                            const DenseTensor& index,
                            DenseTensor src,
                            bool include_self,
                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_kernel(DenseTensor self,
                            int dim,
                            const DenseTensor& index,
                            DenseTensor src,
                            bool include_self,
                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mean_kernel(DenseTensor self,
                             int dim,
                             const DenseTensor& index,
                             DenseTensor src,
                             bool include_self,
                             const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_max_kernel(DenseTensor self,
                            int dim,
                            const DenseTensor& index,
                            DenseTensor src,
                            bool include_self,
                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_min_kernel(DenseTensor self,
                            int dim,
                            const DenseTensor& index,
                            DenseTensor src,
                            bool include_self,
                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_input_grad_kernel(DenseTensor self,
                                   int dim,
                                   const DenseTensor& index,
                                   DenseTensor grad,
                                   bool include_self,
                                   const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_min_max_input_grad_kernel(DenseTensor self,
                                               int dim,
                                               const DenseTensor& index,
                                               const DenseTensor& out,
                                               const DenseTensor& x,
                                               const DenseTensor& value,
                                               DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mean_input_grad_kernel(DenseTensor self,
                                        int dim,
                                        const DenseTensor& index,
                                        DenseTensor grad,
                                        bool include_self,
                                        const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_value_grad_kernel(DenseTensor self,
                                   int dim,
                                   const DenseTensor& index,
                                   DenseTensor grad,
                                   bool include_self,
                                   const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_add_mean_value_grad_kernel(DenseTensor self,
                                            int dim,
                                            const DenseTensor& index,
                                            const DenseTensor& out,
                                            const DenseTensor& x,
                                            const DenseTensor& value,
                                            DenseTensor grad,
                                            const std::string& reduce,
                                            bool include_self,
                                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_min_max_value_grad_kernel(DenseTensor self,
                                               int dim,
                                               const DenseTensor& index,
                                               const DenseTensor& out,
                                               const DenseTensor& x,
                                               const DenseTensor& value,
                                               DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_gather_kernel(DenseTensor self,
                       int dim,
                       const DenseTensor& index,
                       DenseTensor result,
                       bool include_self,
                       const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_assign_kernel(DenseTensor self,
                               int dim,
                               const DenseTensor& index,
                               DenseTensor src,
                               bool include_self,
                               const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_add_kernel(DenseTensor self,
                            int dim,
                            const DenseTensor& index,
                            DenseTensor src,
                            bool include_self,
                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_kernel(DenseTensor self,
                            int dim,
                            const DenseTensor& index,
                            DenseTensor src,
                            bool include_self,
                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mean_kernel(DenseTensor self,
                             int dim,
                             const DenseTensor& index,
                             DenseTensor src,
                             bool include_self,
                             const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_max_kernel(DenseTensor self,
                            int dim,
                            const DenseTensor& index,
                            DenseTensor src,
                            bool include_self,
                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_min_kernel(DenseTensor self,
                            int dim,
                            const DenseTensor& index,
                            DenseTensor src,
                            bool include_self,
                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_input_grad_kernel(DenseTensor self,
                                   int dim,
                                   const DenseTensor& index,
                                   DenseTensor grad,
                                   bool include_self,
                                   const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_min_max_input_grad_kernel(DenseTensor self UNUSED,
                                               int dim,
                                               const DenseTensor& index,
                                               const DenseTensor& out,
                                               const DenseTensor& x,
                                               const DenseTensor& value,
                                               DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mean_input_grad_kernel(DenseTensor self,
                                        int dim,
                                        const DenseTensor& index,
                                        DenseTensor grad,
                                        bool include_self,
                                        const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_value_grad_kernel(DenseTensor self,
                                   int dim,
                                   const DenseTensor& index,
                                   DenseTensor grad,
                                   bool include_self,
                                   const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_add_mean_value_grad_kernel(DenseTensor self,
                                            int dim,
                                            const DenseTensor& index,
                                            const DenseTensor& out,
                                            const DenseTensor& x,
                                            const DenseTensor& value,
                                            DenseTensor grad,
                                            const std::string& reduce,
                                            bool include_self,
                                            const DeviceContext& dev_ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_min_max_value_grad_kernel(DenseTensor self,
                                               int dim,
                                               const DenseTensor& index,
                                               const DenseTensor& out,
                                               const DenseTensor& x,
                                               const DenseTensor& value,
                                               DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const DeviceContext& dev_ctx);

// Brings the operands of ``put_along_axis`` into the representation that the
// gather/scatter functor below can actually address, and raises the diagnosis
// that only the kernel is able to give.
//
// torch represents a 0-D tensor as a 1-D tensor holding a single element
// (``ensure_nonempty_dim`` / ``ensure_nonempty_size`` /
// ``ensure_nonempty_vec``) and restrides self, index and src into that
// representation before entering its scatter kernel, so its shape check and its
// kernel agree on one set of ranks.
// ``PutAlongAxisInferMeta`` follows the same rule. The functor, however,
// indexes
// ``dims()`` and ``strides()`` directly and sizes ``CoordinateManager`` from
// ``index.dims().size()``: a rank-0 operand would read a slot that
// ``calc_strides`` never wrote, and a rank-0 ``index`` would give ``ndim ==
// 0``, leaving ``CoordinateManager::indices`` empty while it is still indexed.
// So the promotion has to happen here as well.
//
// ``Resize`` rewrites the metadata of the tensors passed here and nothing else,
// so a shallow view can be promoted without touching the caller's tensor or the
// underlying buffer. ``self`` is the one operand a kernel may have to promote
// in place, since the scatter writes through it: the ``put_along_axis`` kernels
// pass their output and restore its shape once the scatter is done, so the
// promotion never reaches the caller.
//
// ``axis`` is normalized at the same time. It reaches the kernel exactly as the
// caller wrote it, so it can still be negative when the op is entered through
// ``_C_ops`` rather than the python wrapper.
inline void PreparePutAlongAxisOperands(DenseTensor* self,
                                        DenseTensor* index,
                                        DenseTensor* value,
                                        int* axis) {
  const int rank = std::max<int>(static_cast<int>(self->dims().size()), 1);
  if (*axis < 0) {
    *axis += rank;
  }
  const DDim single_element({1});
  if (self->dims().size() == 0) {
    self->Resize(single_element);
  }
  if (index->dims().size() == 0) {
    index->Resize(single_element);
  }
  if (value->dims().size() == 0) {
    value->Resize(single_element);
  }

  // A 0-size scatter dimension leaves no valid index value at all, so every
  // element of a non-empty ``index`` is out of bounds. torch reports this from
  // the scatter kernel as an index out-of-bounds error rather than from its
  // shape check, so the diagnosis is raised here instead of in the InferMeta.
  //
  // The shared gather/scatter functor cannot do it: it early-returns on any
  // 0-size operand, and that early return is relied upon by the grad kernels of
  // ``take_along_axis``, ``cummax``/``cummin`` and friends, which legitimately
  // scatter a non-empty index into a 0-size buffer.
  if (index->numel() == 0) {
    return;
  }
  PADDLE_ENFORCE_NE(
      self->dims()[*axis],
      0,
      common::errors::OutOfRange(
          "The index is out of bounds, please check whether the index and "
          "input's shape meet the requirements. It should be greater or equal "
          "to [0] and less than [0], which leaves no valid index value at all "
          "because dimension [%d] of the input has size 0.",
          *axis));
}

}  // namespace funcs
}  // namespace phi
