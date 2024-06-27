/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_MUSA
#include "glog/logging.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/gpudnn/matmul_gpudnn.h"
#include "paddle/phi/kernels/impl/matmul_grad_kernel_impl.h"
#include "paddle/phi/kernels/matmul_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MatmulGPUDNNKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        bool transpose_x,
                        bool transpose_y,
                        DenseTensor* out);

static inline int BatchSizeOfTensor(const DenseTensor& t) {
  auto dims = t.dims();
  if (dims.size() <= 2) {
    return 0;
  }
  auto dim_vec = common::vectorize<int>(dims);
  int bs = 1;
  for (size_t i = 0; i < dim_vec.size() - 2; ++i) {
    bs *= dim_vec[i];
  }
  return bs;
}

static inline std::vector<int> FoldedDims(const DenseTensor& x) {
  std::vector<int> origin_dims = common::vectorize<int>(x.dims());
  int ndim = origin_dims.size();
  int batch_size = BatchSizeOfTensor(x);
  int folded_vec_size = ndim > 3 ? 3 : ndim;
  std::vector<int> folded_dims(folded_vec_size);

  std::copy(
      origin_dims.end() - ndim, origin_dims.end(), folded_dims.end() - ndim);
  if (batch_size >= 1) {
    folded_dims.front() = batch_size;
  }

  return folded_dims;
}

template <typename Context, typename T>
typename std::enable_if<!std::is_integral<T>::value>::type MatMulDNN(
    const Context& dev_ctx,
    const DenseTensor& x,
    bool trans_x,
    const DenseTensor& y,
    bool trans_y,
    DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  DenseTensor x_helper =
      paddle::experimental::CheckAndTrans2NewContiguousTensor(x);
  DenseTensor y_helper =
      paddle::experimental::CheckAndTrans2NewContiguousTensor(y);
  int x_batch_size = BatchSizeOfTensor(x_helper);
  int y_batch_size = BatchSizeOfTensor(y_helper);
  int out_batch_size = BatchSizeOfTensor(*out);
  if (x_batch_size >= 1 || y_batch_size >= 1) {
    std::vector<int> folded_x_dims = FoldedDims(x_helper);
    std::vector<int> folded_y_dims = FoldedDims(y_helper);
    std::vector<int> folded_out_dims = FoldedDims(*out);
    phi::BmmGPUDNNKernelImpl<T>(dev_ctx,
                                x_helper.data<T>(),
                                trans_x,
                                folded_x_dims,
                                y_helper.data<T>(),
                                trans_y,
                                folded_y_dims,
                                out->data<T>(),
                                folded_out_dims);

  } else {
    phi::MatMulGPUDNNKernelImpl<T, Context>(
        dev_ctx, x_helper, trans_x, y_helper, trans_y, out);
  }
}

template <typename T, typename Context>
void CalcInputGradDNN(const Context& dev_ctx,
                      const DenseTensor& x,
                      bool trans_x,
                      bool is_fold_init_dims_x,
                      const DenseTensor& y,
                      bool trans_y,
                      bool is_fold_init_dims_y,
                      DenseTensor* out) {
  if (out == nullptr) {
    return;
  }
  bool need_combine =
      (x.dims().size() == 3 || y.dims().size() == 3) && out->dims().size() == 2;
  if (!need_combine) {
    MatMulDNN<Context, T>(dev_ctx, x, trans_x, y, trans_y, out);
  } else {
    // fold first, otherwise may need to do reduce-sum on out
    MatMulDNN<Context, T>(
        dev_ctx,
        is_fold_init_dims_x ? FoldInitDims(x)
                            : FoldHeadAndLastDims<Context, T>(dev_ctx, x),
        trans_x,
        is_fold_init_dims_y ? FoldInitDims(y)
                            : FoldHeadAndLastDims<Context, T>(dev_ctx, y),
        trans_y,
        out);
  }
}

template <typename T, typename Context>
void MatmulGradGPUDNNKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            const DenseTensor& out_grad,
                            bool transpose_x,
                            bool transpose_y,
                            DenseTensor* dx,
                            DenseTensor* dy) {
  std::vector<std::int64_t> x_dims = common::vectorize(x.dims());
  std::vector<std::int64_t> y_dims = common::vectorize(y.dims());
  std::vector<std::int64_t> dout_dims = common::vectorize(out_grad.dims());

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int ndim = dout_dims.size();

  if (x_ndim == 1 && y_ndim == 1) {
    if (dx) dev_ctx.template Alloc<T>(dx);
    if (dy) dev_ctx.template Alloc<T>(dy);
    if (out_grad.numel() == 1) {
      // any other cases that out_grad.numel() != 1 ?
      DotGradFunction<Context, T>()(dev_ctx, &x, &y, &out_grad, dx, dy);
      return;
    }
  }

  bool is_broadcast = true;
  if (x_ndim <= 2 || y_ndim <= 2) {
    is_broadcast = false;
  } else if (x_ndim != y_ndim) {
    is_broadcast = true;
  } else {
    is_broadcast = !std::equal(
        x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2, y_dims.cbegin());
  }

  if (!is_broadcast) {
    DenseTensor x_help = x;
    DenseTensor y_help = y;
    DenseTensor out_grad_help = out_grad;

    ReshapeXYOutIntoMatrixSequence(
        &x_help, &y_help, &out_grad_help, transpose_x, transpose_y);

    DDim dx_dims, dy_dims;
    if (dx) {
      dx_dims = dx->dims();
      if (dx_dims != x_help.dims()) {
        dx->Resize(x_help.dims());
      }
    }
    if (dy) {
      dy_dims = dy->dims();
      if (dy_dims != y_help.dims()) {
        dy->Resize(y_help.dims());
      }
    }

    if (transpose_x && transpose_y) {
      CalcInputGradDNN<T>(
          dev_ctx, y_help, true, true, out_grad_help, true, false, dx);
      CalcInputGradDNN<T>(
          dev_ctx, out_grad_help, true, true, x_help, true, false, dy);
    } else if (transpose_x) {
      CalcInputGradDNN<T>(
          dev_ctx, y_help, false, false, out_grad_help, true, false, dx);
      CalcInputGradDNN<T>(
          dev_ctx, x_help, false, false, out_grad_help, false, true, dy);
    } else if (transpose_y) {
      CalcInputGradDNN<T>(
          dev_ctx, out_grad_help, false, false, y_help, false, true, dx);
      CalcInputGradDNN<T>(
          dev_ctx, out_grad_help, true, true, x_help, false, true, dy);
    } else {
      CalcInputGradDNN<T>(
          dev_ctx, out_grad_help, false, false, y_help, true, false, dx);
      CalcInputGradDNN<T>(
          dev_ctx, x_help, true, true, out_grad_help, false, true, dy);
    }

    if (dx) {
      if (dx_dims != x_help.dims()) {
        dx->Resize(dx_dims);
      }
    }
    if (dy) {
      if (dy_dims != y_help.dims()) {
        dy->Resize(dy_dims);
      }
    }
  } else {
    // PADDLE_THROW(phi::errors::Unimplemented(
    //     "broadcast case not supported in MatmulGradGPUDNNKernel"));
    VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
               "wastes the memory. So we should avoid the case in reality";

    DenseTensor dx_help;
    DenseTensor dy_help;

    if (transpose_x && transpose_y) {
      // X'Y': dX = Y'G', dY = G'X'
      if (dx) {
        MatmulGPUDNNKernel<T, Context>(
            dev_ctx, y, out_grad, true, true, &dx_help);
      }
      if (dy) {
        MatmulGPUDNNKernel<T, Context>(
            dev_ctx, out_grad, x, true, true, &dy_help);
      }
    } else if (transpose_x) {
      // X'Y: dX = YG', dY = XG
      if (dx) {
        MatmulGPUDNNKernel<T, Context>(
            dev_ctx, y, out_grad, false, true, &dx_help);
      }
      if (dy) {
        MatmulGPUDNNKernel<T, Context>(
            dev_ctx, x, out_grad, false, false, &dy_help);
      }
    } else if (transpose_y) {
      // XY': dX = GY, dY = G'X
      if (dx) {
        MatmulGPUDNNKernel<T, Context>(
            dev_ctx, out_grad, y, false, false, &dx_help);
      }
      if (dy) {
        MatmulGPUDNNKernel<T, Context>(
            dev_ctx, out_grad, x, true, false, &dy_help);
      }
    } else {
      // XY: dX = GY', dY = X'G
      if (dx) {
        MatmulGPUDNNKernel<T, Context>(
            dev_ctx, out_grad, y, false, true, &dx_help);
      }
      if (dy) {
        MatmulGPUDNNKernel<T, Context>(
            dev_ctx, x, out_grad, true, false, &dy_help);
      }
    }

    const std::vector<std::int64_t> dx_help_dims =
        common::vectorize(dx_help.dims());
    const std::vector<std::int64_t> dy_help_dims =
        common::vectorize(dy_help.dims());

    std::vector<std::int64_t> dx_broadcast_dims(ndim);
    std::vector<std::int64_t> dy_broadcast_dims(ndim);

    std::fill(
        dx_broadcast_dims.data(), dx_broadcast_dims.data() + ndim - x_ndim, 1);
    std::fill(
        dy_broadcast_dims.data(), dy_broadcast_dims.data() + ndim - y_ndim, 1);
    std::copy(x_dims.data(),
              x_dims.data() + x_ndim,
              dx_broadcast_dims.data() + ndim - x_ndim);
    std::copy(y_dims.data(),
              y_dims.data() + y_ndim,
              dy_broadcast_dims.data() + ndim - y_ndim);

    std::vector<int> dx_reduce_dims;
    std::vector<int> dy_reduce_dims;
    for (int idx = 0; idx <= ndim - 3; idx++) {
      if (dx_help_dims[idx] != 1 && dx_broadcast_dims[idx] == 1) {
        dx_reduce_dims.push_back(idx);
      }
      if (dy_help_dims[idx] != 1 && dy_broadcast_dims[idx] == 1) {
        dy_reduce_dims.push_back(idx);
      }
    }

    // reduce sum
    if (dx) {
      if (dx_reduce_dims.empty()) {
        *dx = std::move(dx_help);
      } else {
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, dx_help, dx, dx_reduce_dims);
      }
      dx->Resize(x.dims());
    }
    if (dy) {
      if (dy_reduce_dims.empty()) {
        *dy = std::move(dy_help);
      } else {
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, dy_help, dy, dy_reduce_dims);
      }
      dy->Resize(y.dims());
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(matmul_grad,  // musa_only
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::MatmulGradGPUDNNKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
