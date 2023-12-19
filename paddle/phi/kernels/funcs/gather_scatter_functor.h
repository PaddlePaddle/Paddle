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

#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

#pragma once

namespace phi {
namespace funcs {

#define Instantiate_Template_Function(func)                                  \
  Instantiate_Template_Function_index_t(                                     \
      func, int) Instantiate_Template_Function_index_t(func, float)          \
      Instantiate_Template_Function_index_t(                                 \
          func, double) Instantiate_Template_Function_index_t(func, int64_t) \
          Instantiate_Template_Function_index_t(func, phi::dtype::float16)   \
              Instantiate_Template_Function_index_t(func,                    \
                                                    phi::dtype::bfloat16)    \
                  Instantiate_Template_Function_index_t(func, unsigned char)

#define Instantiate_Template_Function_index_t(func, tensor_t)          \
  template void func<tensor_t, int>(phi::DenseTensor input,            \
                                    int dim,                           \
                                    const phi::DenseTensor& index,     \
                                    phi::DenseTensor result,           \
                                    bool include_self,                 \
                                    const phi::DeviceContext& ctx);    \
  template void func<tensor_t, int64_t>(phi::DenseTensor input,        \
                                        int dim,                       \
                                        const phi::DenseTensor& index, \
                                        phi::DenseTensor result,       \
                                        bool include_self,             \
                                        const phi::DeviceContext& ctx);

#define Instantiate_Template_Function_With_Out(func)                        \
  Instantiate_Template_Function_index_t_With_Out(func, int)                 \
      Instantiate_Template_Function_index_t_With_Out(func, float)           \
          Instantiate_Template_Function_index_t_With_Out(func, double)      \
              Instantiate_Template_Function_index_t_With_Out(func, int64_t) \
                  Instantiate_Template_Function_index_t_With_Out(           \
                      func, phi::dtype::float16)                            \
                      Instantiate_Template_Function_index_t_With_Out(       \
                          func, phi::dtype::bfloat16)                       \
                          Instantiate_Template_Function_index_t_With_Out(   \
                              func, unsigned char)
#define Instantiate_Template_Function_index_t_With_Out(func, tensor_t) \
  template void func<tensor_t, int>(phi::DenseTensor input,            \
                                    int dim,                           \
                                    const phi::DenseTensor& index,     \
                                    const phi::DenseTensor& out,       \
                                    const phi::DenseTensor& self,      \
                                    const phi::DenseTensor& value,     \
                                    phi::DenseTensor result,           \
                                    const std::string& reduce,         \
                                    bool include_self,                 \
                                    const phi::DeviceContext& ctx);    \
  template void func<tensor_t, int64_t>(phi::DenseTensor input,        \
                                        int dim,                       \
                                        const phi::DenseTensor& index, \
                                        const phi::DenseTensor& out,   \
                                        const phi::DenseTensor& self,  \
                                        const phi::DenseTensor& value, \
                                        phi::DenseTensor result,       \
                                        const std::string& reduce,     \
                                        bool include_self,             \
                                        const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_gather_kernel(phi::DenseTensor self,
                       int dim,
                       const phi::DenseTensor& index,
                       phi::DenseTensor result,
                       bool include_self,
                       const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_assign_kernel(phi::DenseTensor self,
                               int dim,
                               const phi::DenseTensor& index,
                               phi::DenseTensor src,
                               bool include_self,
                               const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_add_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mean_kernel(phi::DenseTensor self,
                             int dim,
                             const phi::DenseTensor& index,
                             phi::DenseTensor src,
                             bool include_self,
                             const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_max_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_min_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_input_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self,
                                   const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_min_max_input_grad_kernel(phi::DenseTensor self,
                                               int dim,
                                               const phi::DenseTensor& index,
                                               const phi::DenseTensor& out,
                                               const phi::DenseTensor& x,
                                               const phi::DenseTensor& value,
                                               phi::DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mean_input_grad_kernel(phi::DenseTensor self,
                                        int dim,
                                        const phi::DenseTensor& index,
                                        phi::DenseTensor grad,
                                        bool include_self,
                                        const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_value_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self,
                                   const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_add_mean_value_grad_kernel(phi::DenseTensor self,
                                            int dim,
                                            const phi::DenseTensor& index,
                                            const phi::DenseTensor& out,
                                            const phi::DenseTensor& x,
                                            const phi::DenseTensor& value,
                                            phi::DenseTensor grad,
                                            const std::string& reduce,
                                            bool include_self,
                                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_min_max_value_grad_kernel(phi::DenseTensor self,
                                               int dim,
                                               const phi::DenseTensor& index,
                                               const phi::DenseTensor& out,
                                               const phi::DenseTensor& x,
                                               const phi::DenseTensor& value,
                                               phi::DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_gather_kernel(phi::DenseTensor self,
                       int dim,
                       const phi::DenseTensor& index,
                       phi::DenseTensor result,
                       bool include_self,
                       const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_assign_kernel(phi::DenseTensor self,
                               int dim,
                               const phi::DenseTensor& index,
                               phi::DenseTensor src,
                               bool include_self,
                               const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_add_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mean_kernel(phi::DenseTensor self,
                             int dim,
                             const phi::DenseTensor& index,
                             phi::DenseTensor src,
                             bool include_self,
                             const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_max_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_min_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_input_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self,
                                   const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_min_max_input_grad_kernel(phi::DenseTensor self UNUSED,
                                               int dim,
                                               const phi::DenseTensor& index,
                                               const phi::DenseTensor& out,
                                               const phi::DenseTensor& x,
                                               const phi::DenseTensor& value,
                                               phi::DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mean_input_grad_kernel(phi::DenseTensor self,
                                        int dim,
                                        const phi::DenseTensor& index,
                                        phi::DenseTensor grad,
                                        bool include_self,
                                        const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_value_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self,
                                   const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_add_mean_value_grad_kernel(phi::DenseTensor self,
                                            int dim,
                                            const phi::DenseTensor& index,
                                            const phi::DenseTensor& out,
                                            const phi::DenseTensor& x,
                                            const phi::DenseTensor& value,
                                            phi::DenseTensor grad,
                                            const std::string& reduce,
                                            bool include_self,
                                            const phi::DeviceContext& ctx);

template <typename tensor_t, typename index_t>
void gpu_scatter_mul_min_max_value_grad_kernel(phi::DenseTensor self,
                                               int dim,
                                               const phi::DenseTensor& index,
                                               const phi::DenseTensor& out,
                                               const phi::DenseTensor& x,
                                               const phi::DenseTensor& value,
                                               phi::DenseTensor grad,
                                               const std::string& reduce,
                                               bool include_self,
                                               const phi::DeviceContext& ctx);

}  // namespace funcs
}  // namespace phi
