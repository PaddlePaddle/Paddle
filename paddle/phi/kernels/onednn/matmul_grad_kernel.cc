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

#include "paddle/phi/kernels/matmul_grad_kernel.h"

#include "paddle/phi/backends/onednn/matmul_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

void CalculateMatrixDims(const std::vector<int64_t> &x_dims,
                         const std::vector<int64_t> &y_dims,
                         const std::vector<int64_t> &out_dims,
                         std::vector<int64_t> *x_bd_dims,
                         std::vector<int64_t> *y_bd_dims,
                         std::vector<int64_t> *out_bd_dims,
                         bool trans_x,
                         bool trans_y) {
  if (x_dims.size() == 1) {
    (*x_bd_dims)[x_bd_dims->size() - 1] = x_dims[0];
  } else if (x_dims.size() == 2) {
    (*x_bd_dims)[x_bd_dims->size() - 1] = x_dims[1];
    (*x_bd_dims)[x_bd_dims->size() - 2] = x_dims[0];
  } else {
    for (size_t i = 0; i < x_dims.size(); ++i) {
      (*x_bd_dims)[x_bd_dims->size() - x_dims.size() + i] = x_dims[i];
    }
  }
  if (y_dims.size() == 1) {
    (*y_bd_dims)[x_bd_dims->size() - 2] = y_dims[0];
  } else if (y_dims.size() == 2) {
    (*y_bd_dims)[y_bd_dims->size() - 1] = y_dims[1];
    (*y_bd_dims)[y_bd_dims->size() - 2] = y_dims[0];
  } else {
    for (size_t i = 0; i < y_dims.size(); ++i) {
      (*y_bd_dims)[y_bd_dims->size() - y_dims.size() + i] = y_dims[i];
    }
  }

  for (size_t i = 0; i < x_bd_dims->size() - 2; ++i) {
    (*out_bd_dims)[i] = std::max((*x_bd_dims)[i], (*y_bd_dims)[i]);
  }
  int h_idx = trans_x ? x_bd_dims->size() - 1 : x_bd_dims->size() - 2;
  int w_idx = trans_y ? y_bd_dims->size() - 2 : y_bd_dims->size() - 1;

  (*out_bd_dims)[x_bd_dims->size() - 2] = (*x_bd_dims)[h_idx];
  (*out_bd_dims)[y_bd_dims->size() - 1] = (*y_bd_dims)[w_idx];
}

template <typename T>
void CalculateGradMatrixDims(const OneDNNContext &dev_ctx,
                             DenseTensor *dx_tmp,
                             DenseTensor *dy_tmp,
                             std::vector<int64_t> *dx_bd_dims,
                             std::vector<int64_t> *dy_bd_dims) {
  for (size_t i = 0; i < dx_bd_dims->size() - 2; ++i) {
    if ((*dx_bd_dims)[i] != (*dy_bd_dims)[i]) {
      if ((*dx_bd_dims)[i] == 1) {
        (*dx_bd_dims)[i] = (*dy_bd_dims)[i];
      } else {
        (*dy_bd_dims)[i] = (*dx_bd_dims)[i];
      }
    }
  }

  dx_tmp->Resize(make_ddim(*dx_bd_dims));
  dev_ctx.template Alloc<T>(dx_tmp);
  dy_tmp->Resize(make_ddim(*dy_bd_dims));
  dev_ctx.template Alloc<T>(dy_tmp);
}

template <typename T>
void ReduceSumForMatmulGradOutput(const OneDNNContext &dev_ctx,
                                  const DenseTensor *dx_tmp,
                                  DenseTensor *dx,
                                  const std::vector<int64_t> &dx_dims,
                                  const std::vector<int64_t> &x_dims) {
  funcs::ReductionOneDNNHandler<T> handler(dnnl::algorithm::reduction_sum,
                                           0.0f,
                                           0.0f,
                                           dev_ctx.GetEngine(),
                                           dev_ctx.GetPlace(),
                                           dx_tmp,
                                           dx,
                                           x_dims);

  auto src_memory_p = handler.AcquireSrcMemory(dx_tmp);
  auto dst_memory_p = handler.AcquireDstMemory(dx);

  std::unordered_map<int, dnnl::memory> reduction_args = {
      {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

  auto &astream = OneDNNContext::tls().get_stream();
  auto reduction_p = handler.AcquireForwardPrimitive();

  reduction_p->execute(astream, reduction_args);
  astream.wait();
}

template <typename T, typename Context>
void MatmulGradKernel(const Context &dev_ctx,
                      const DenseTensor &x,
                      const DenseTensor &y,
                      const DenseTensor &dout,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor *dx,
                      DenseTensor *dy) {
  auto x_dims = vectorize(x.dims());
  auto y_dims = vectorize(y.dims());
  auto dout_dims = vectorize(dout.dims());

  size_t ndims = std::max(x_dims.size(), y_dims.size());
  ndims = std::max<size_t>(ndims, 3);

  // in broadcasting scenario new memory is required because
  // reduce sum must be calculated upon broadcasted dims
  DenseTensor dx_tmp, dy_tmp;
  std::vector<int64_t> dout_bd_dims(ndims, 1);
  std::vector<int64_t> x_bd_dims(ndims, 1);
  std::vector<int64_t> y_bd_dims(ndims, 1);

  CalculateMatrixDims(x_dims,
                      y_dims,
                      dout_dims,
                      &x_bd_dims,
                      &y_bd_dims,
                      &dout_bd_dims,
                      transpose_x,
                      transpose_y);

  std::vector<int64_t> dx_bd_dims(x_bd_dims);
  std::vector<int64_t> dy_bd_dims(y_bd_dims);

  CalculateGradMatrixDims<T>(
      dev_ctx, &dx_tmp, &dy_tmp, &dx_bd_dims, &dy_bd_dims);

  if (transpose_x && transpose_y) {
    funcs::ExecuteMatmul<T, T>(
        dev_ctx, y, dout, y_bd_dims, dout_bd_dims, true, true, &dx_tmp);
    funcs::ExecuteMatmul<T, T>(
        dev_ctx, dout, x, dout_bd_dims, x_bd_dims, true, true, &dy_tmp);
  } else if (transpose_x) {
    funcs::ExecuteMatmul<T, T>(
        dev_ctx, y, dout, y_bd_dims, dout_bd_dims, false, true, &dx_tmp);
    funcs::ExecuteMatmul<T, T>(
        dev_ctx, x, dout, x_bd_dims, dout_bd_dims, false, false, &dy_tmp);
  } else if (transpose_y) {
    funcs::ExecuteMatmul<T, T>(
        dev_ctx, dout, y, dout_bd_dims, y_bd_dims, false, false, &dx_tmp);
    funcs::ExecuteMatmul<T, T>(
        dev_ctx, dout, x, dout_bd_dims, x_bd_dims, true, false, &dy_tmp);
  } else {
    funcs::ExecuteMatmul<T, T>(
        dev_ctx, dout, y, dout_bd_dims, y_bd_dims, false, true, &dx_tmp);
    funcs::ExecuteMatmul<T, T>(
        dev_ctx, x, dout, x_bd_dims, dout_bd_dims, true, false, &dy_tmp);
  }

  if (x_bd_dims != dx_bd_dims) {
    ReduceSumForMatmulGradOutput<T>(
        dev_ctx, &dx_tmp, dx, dx_bd_dims, x_bd_dims);
  } else {
    *dx = std::move(dx_tmp);
  }
  if (y_bd_dims != dy_bd_dims) {
    ReduceSumForMatmulGradOutput<T>(
        dev_ctx, &dy_tmp, dy, dy_bd_dims, y_bd_dims);
  } else {
    *dy = std::move(dy_tmp);
  }

  dx->set_mem_desc(x.mem_desc());
  dx->Resize(x.dims());
  dy->set_mem_desc(y.mem_desc());
  dy->Resize(y.dims());
}

template <typename T, typename Context>
void MatmulWithFlattenGradKernel(const Context &dev_ctx,
                                 const DenseTensor &x,
                                 const DenseTensor &y,
                                 const DenseTensor &out_grad,
                                 int x_num_col_dims,
                                 int y_num_col_dims,
                                 DenseTensor *x_grad,
                                 DenseTensor *y_grad) {
  const DenseTensor reshaped_y = phi::ReshapeToMatrix(y, y_num_col_dims);
  const DenseTensor reshaped_x = phi::ReshapeToMatrix(x, x_num_col_dims);
  const DenseTensor x_matrix = x.dims().size() > 2 ? reshaped_x : x;
  const DenseTensor y_matrix = y.dims().size() > 2 ? reshaped_y : y;

  DenseTensor dout_matrix = out_grad;
  dout_matrix.Resize({flatten_to_2d(x.dims(), x_num_col_dims)[0],
                      flatten_to_2d(y.dims(), y_num_col_dims)[1]});

  // adding mb dim because MatMulV2 handler needs it
  std::vector<int64_t> x_dims(3, 1);
  std::vector<int64_t> y_dims(3, 1);
  std::vector<int64_t> dout_dims(3, 1);
  x_dims[1] = x_matrix.dims()[0];
  x_dims[2] = x_matrix.dims()[1];
  y_dims[1] = y_matrix.dims()[0];
  y_dims[2] = y_matrix.dims()[1];
  dout_dims[1] = dout_matrix.dims()[0];
  dout_dims[2] = dout_matrix.dims()[1];

  if (x_grad != nullptr) {
    x_grad->set_lod(x.lod());
    funcs::ExecuteMul<T>(
        dev_ctx, dout_matrix, y_matrix, dout_dims, y_dims, false, true, x_grad);
  }
  if (y_grad != nullptr) {
    y_grad->set_lod(y.lod());
    funcs::ExecuteMul<T>(
        dev_ctx, x_matrix, dout_matrix, x_dims, dout_dims, true, false, y_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(matmul_grad,
                   OneDNN,
                   ONEDNN,
                   phi::MatmulGradKernel,
                   float,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(matmul_with_flatten_grad,
                   OneDNN,
                   ONEDNN,
                   phi::MatmulWithFlattenGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
