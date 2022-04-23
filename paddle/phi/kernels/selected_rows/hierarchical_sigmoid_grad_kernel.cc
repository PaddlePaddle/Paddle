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

#include "paddle/phi/kernels/selected_rows/hierarchical_sigmoid_grad_kernel.h"

#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/hierarchical_sigmoid_grad.h"

namespace phi {
namespace sr {

static std::vector<int64_t> PathToRows(const DenseTensor& path) {
  std::set<int64_t> rows;
  const int64_t* paths = path.data<int64_t>();
  for (int64_t i = 0; i < path.numel(); ++i) {
    int64_t row = paths[i];
    if (row < 0) {
      continue;
    }
    rows.emplace(row);
  }
  return std::vector<int64_t>(rows.begin(), rows.end());
}

template <typename T, typename Context>
void HierarchicalSigmoidGradKernel(const Context& ctx,
                                   const DenseTensor& x,
                                   const DenseTensor& w,
                                   const DenseTensor& label,
                                   paddle::optional<const DenseTensor&> path,
                                   paddle::optional<const DenseTensor&> code,
                                   paddle::optional<const DenseTensor&> bias,
                                   const DenseTensor& pre_out,
                                   const DenseTensor& out_grad,
                                   int num_classes,
                                   bool remote_prefetch,
                                   int trainer_id,
                                   const std::vector<int64_t>& height_sections,
                                   const std::vector<std::string>& epmap,
                                   const std::vector<std::string>& table_names,
                                   bool is_sparse,
                                   DenseTensor* x_grad,
                                   SelectedRows* w_grad,
                                   DenseTensor* bias_grad) {
  PADDLE_ENFORCE_NOT_NULL(
      path.get_ptr(),
      errors::NotFound("Custom tree must be set for sparse mode!"));
  paddle::framework::Vector<int64_t> real_rows = PathToRows(*path);
  w_grad->set_rows(real_rows);
  // Build a map of id -> row_index to speed up finding the index of one id
  w_grad->set_height(w.dims()[0]);
  auto* w_grad_value = w_grad->mutable_value();
  phi::DDim temp_dim(w.dims());
  temp_dim[0] = real_rows.size();
  w_grad_value->Resize(temp_dim);
  phi::HierarchicalSigmoidGradKernelImpl<T>(ctx,
                                            x,
                                            w,
                                            label,
                                            path,
                                            code,
                                            bias,
                                            pre_out,
                                            out_grad,
                                            num_classes,
                                            remote_prefetch,
                                            trainer_id,
                                            height_sections,
                                            epmap,
                                            table_names,
                                            is_sparse,
                                            x_grad,
                                            w_grad_value,
                                            bias_grad,
                                            w_grad);
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(hierarchical_sigmoid_grad_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::HierarchicalSigmoidGradKernel,
                   float,
                   double) {}
