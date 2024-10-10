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
#include "paddle/phi/kernels/gpudnn/matmul_gpudnn.h"
#include "paddle/phi/kernels/matmul_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

static void GetBroadcastFromDims(const int x_ndim,
                                 const std::int64_t* x_dims,
                                 const int y_ndim,
                                 const std::int64_t* y_dims,
                                 std::int64_t* x_bd_dims,
                                 std::int64_t* y_bd_dims,
                                 std::int64_t* out_bd_dims) {
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::fill(x_bd_dims, x_bd_dims + ndim - x_ndim, 1);
  std::fill(y_bd_dims, y_bd_dims + ndim - y_ndim, 1);
  std::copy(x_dims, x_dims + x_ndim, x_bd_dims + ndim - x_ndim);
  std::copy(y_dims, y_dims + y_ndim, y_bd_dims + ndim - y_ndim);

  for (int i = 0; i < ndim; i++) {
    PADDLE_ENFORCE_EQ(
        x_bd_dims[i] == y_bd_dims[i] || x_bd_dims[i] <= 1 || y_bd_dims[i] <= 1,
        true,
        phi::errors::InvalidArgument(
            "Input(X) and Input(Y) has error dim. "
            "X_broadcast's shape[%s] must be equal to Y_broadcast's shape[%s], "
            "or X_broadcast's shape[%s] <= 1, or Y_broadcast's shape[%s] <= 1, "
            "but received X_broadcast's shape[%s] = [%s]"
            "received Y_broadcast's shape[%s] = [%s].",
            i,
            i,
            i,
            i,
            i,
            x_bd_dims[i],
            i,
            y_bd_dims[i]));
    if (x_bd_dims[i] == 0 || y_bd_dims[i] == 0) {
      out_bd_dims[i] = 0;
    } else {
      out_bd_dims[i] = (std::max)(x_bd_dims[i], y_bd_dims[i]);
    }
  }
}

template <typename T, typename Context>
void MatmulGPUDNNKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        bool trans_x,
                        bool trans_y,
                        DenseTensor* out) {
  PADDLE_ENFORCE_NE(
      common::product(x.dims()),
      0,
      phi::errors::InvalidArgument("The Input(X) dims size must not be equal 0,"
                                   " but reviced dims size is 0. "));
  PADDLE_ENFORCE_NE(
      common::product(y.dims()),
      0,
      phi::errors::InvalidArgument("The Input(Y) dims size must not be equal 0,"
                                   " but reviced dims size is 0. "));
  const std::vector<int64_t> x_dims = common::vectorize(x.dims());
  const std::vector<int64_t> y_dims = common::vectorize(y.dims());
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  // x_ndim == 1 && y_ndim == 1 ||
  // x_ndim == 1 && y_ndim >= 2 ||
  // y_ndim == 1 && x_ndim >= 2 ||
  // x_ndim >= 2 && y_ndim >= 2 ||
  if (x_ndim == 1 && y_ndim == 1) {
    const int M = x.numel();
    const int N = y.numel();
    PADDLE_ENFORCE_EQ(
        M,
        N,
        phi::errors::InvalidArgument(
            "x's numbers must be equal to y's numbers, "
            "when x/y's dims =1. But received x has [%d] elements, "
            "received y has [%d] elements.",
            M,
            N));
    VLOG(3) << "Matmul's case 1";
    out->Resize(common::make_ddim({1}));
    dev_ctx.template Alloc<T>(out);
    phi::MatMulGPUDNNKernelImpl<T>(dev_ctx, x, trans_x, y, trans_y, out);
    out->Resize(common::make_ddim({}));
    return;
  }

  const int ndim = std::max(x_ndim, y_ndim);
  const int batch_dim = ndim - 2;

  if (x_ndim == 1) {
    // x_ndim == 1 && y_ndim >= 2
    const int K = x.numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1],
          K,
          phi::errors::InvalidArgument("Input(Y) has error dim. "
                                       "Y'dims[%d] must be equal to %d, "
                                       "but received Y'dims[%d] is %d.",
                                       y_ndim - 1,
                                       K,
                                       y_ndim - 1,
                                       y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2],
          K,
          phi::errors::InvalidArgument("Input(Y) has error dim. "
                                       "Y'dims[%d] must be equal to %d, "
                                       "but received Y'dims[%d] is %d.",
                                       y_ndim - 2,
                                       K,
                                       y_ndim - 2,
                                       y_dims[y_ndim - 2]));
    }
    if (y_ndim == 2) {
      VLOG(3) << "Matmul's case 2";
      std::vector<std::int64_t> out_dims(1);
      out_dims.back() = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
      out->ResizeAndAllocate(common::make_ddim(out_dims));
      dev_ctx.template Alloc<T>(out);
      phi::MatMulGPUDNNKernelImpl<T>(dev_ctx, x, trans_x, y, trans_y, out);
      return;
    } else {
      // y_ndim >= 3, y could be folded, then call mv/mm
      VLOG(3) << "Matmul's case 3";
      // C = XY, C' = Y'X'
      if (trans_y) {
        // no extra memory copy
        const std::int32_t folded_dim_y =
            std::accumulate(y_dims.cbegin(),
                            y_dims.cend() - 1,
                            static_cast<int32_t>(1),
                            std::multiplies<std::int32_t>());
        std::vector<int> folded_y_dims = {folded_dim_y,
                                          (int)y_dims.back()};  // NOLINT
        std::vector<int> x_tmp_dims = {K};
        std::vector<int> out_tmp_dims = {folded_dim_y};

        std::vector<int> out_dims(y_dims.begin(), y_dims.end() - 1);
        out->ResizeAndAllocate(common::make_ddim(out_dims));
        dev_ctx.template Alloc<T>(out);
        phi::MatMulGPUDNNKernelImpl<T>(dev_ctx,
                                       y,
                                       false,  // no trans
                                       folded_y_dims,
                                       x,
                                       false,  // x_ndim == 1
                                       x_tmp_dims,
                                       out,
                                       out_tmp_dims);
      } else {
        const std::int64_t y_batch_size =
            std::accumulate(y_dims.cbegin(),
                            y_dims.cbegin() + batch_dim,
                            1LL,
                            std::multiplies<std::int64_t>());
        std::vector<int> xx_dims = {1, K};
        std::vector<int> maybe_squeezed_y_dims = {
            static_cast<int>(y_batch_size),
            static_cast<int>(y_dims[y_ndim - 2]),
            static_cast<int>(y_dims[y_ndim - 1])};
        std::vector<int> out_dims(y_ndim - 1);
        std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
        out_dims.back() = y_dims.back();

        std::vector<int> maybe_squeezed_out_dims = {
            static_cast<int>(y_batch_size),
            1,
            static_cast<int>(out_dims.back())};
        out->ResizeAndAllocate(common::make_ddim(out_dims));
        dev_ctx.template Alloc<T>(out);
        phi::BmmGPUDNNKernelImpl<T>(dev_ctx,
                                    x,
                                    trans_x,
                                    xx_dims,
                                    y,
                                    trans_y,
                                    maybe_squeezed_y_dims,
                                    out,
                                    maybe_squeezed_out_dims);
      }
      return;
    }
  }

  if (y_ndim == 1) {
    // y_ndim == 1 && x_ndim >= 3
    const int N = y.numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 2],
          N,
          phi::errors::InvalidArgument("Input(X) has error dim."
                                       "X'dims[%d] must be equal to %d"
                                       "But received X'dims[%d] is %d",
                                       x_ndim - 2,
                                       N,
                                       x_ndim - 2,
                                       x_dims[x_ndim - 2]));
    } else {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 1],
          N,
          phi::errors::InvalidArgument("Input(X) has error dim."
                                       "X'dims[%d] must be equal to %d"
                                       "But received X'dims[%d] is %d",
                                       x_ndim - 1,
                                       N,
                                       x_ndim - 1,
                                       x_dims[x_ndim - 1]));
    }
    if (x_ndim == 2) {
      VLOG(3) << "Matmul's case 4";
      std::vector<std::int64_t> out_dims(1);
      out_dims.back() = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
      out->ResizeAndAllocate(common::make_ddim(out_dims));
      dev_ctx.template Alloc<T>(out);
      phi::MatMulGPUDNNKernelImpl<T>(dev_ctx, x, trans_x, y, trans_y, out);
      return;
    } else {
      // x_ndim >= 3, x could be folded, then call mv/mm
      VLOG(3) << "Matmul's case 5";
      if (trans_x) {
        const std::int64_t x_batch_size =
            std::accumulate(x_dims.cbegin(),
                            x_dims.cbegin() + batch_dim,
                            1LL,
                            std::multiplies<std::int64_t>());
        std::vector<int> yy_dims = {N, 1};
        std::vector<int> maybe_squeezed_x_dims = {
            static_cast<int>(x_batch_size),
            static_cast<int>(x_dims[x_ndim - 2]),
            static_cast<int>(x_dims[x_ndim - 1])};
        std::vector<int> out_dims(x_ndim - 1);
        std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
        out_dims.back() = x_dims.back();

        std::vector<int> maybe_squeezed_out_dims = {
            (int)x_batch_size, (int)out_dims.back(), 1};  // NOLINT
        out->ResizeAndAllocate(common::make_ddim(out_dims));
        dev_ctx.template Alloc<T>(out);
        phi::BmmGPUDNNKernelImpl<T>(dev_ctx,
                                    x,
                                    trans_x,
                                    maybe_squeezed_x_dims,
                                    y,
                                    trans_y,
                                    yy_dims,
                                    out,
                                    maybe_squeezed_out_dims);
      } else {
        // !trans_x
        const std::int32_t folded_dim_x =
            std::accumulate(x_dims.cbegin(),
                            x_dims.cend() - 1,
                            static_cast<int32_t>(1),
                            std::multiplies<std::int32_t>());
        std::vector<int> folded_x_dims = {folded_dim_x,
                                          (int)x_dims.back()};  // NOLINT
        std::vector<int> y_tmp_dims = {N};
        std::vector<int> out_tmp_dims = {folded_dim_x};

        std::vector<int> out_dims(x_dims.begin(), x_dims.end() - 1);
        out->ResizeAndAllocate(common::make_ddim(out_dims));
        dev_ctx.template Alloc<T>(out);
        phi::MatMulGPUDNNKernelImpl<T>(dev_ctx,
                                       x,
                                       false,
                                       folded_x_dims,
                                       y,
                                       false,
                                       y_tmp_dims,
                                       out,
                                       out_tmp_dims);
      }
      return;
    }
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 1],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim. "
                                     "Y'dims[%d] must be equal to %d, "
                                     "but received Y'dims[%d] is %d.",
                                     y_ndim - 1,
                                     K,
                                     y_ndim - 1,
                                     y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 2],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim. "
                                     "Y'dims[%d] must be equal to %d, "
                                     "but received Y'dims[%d] is %d.",
                                     y_ndim - 2,
                                     K,
                                     y_ndim - 2,
                                     y_dims[y_ndim - 2]));
  }

  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  std::vector<std::int64_t> x_broadcast_dims(ndim);
  std::vector<std::int64_t> y_broadcast_dims(ndim);
  std::vector<std::int64_t> out_broadcast_dims(ndim);

  GetBroadcastFromDims(x_ndim - 2,
                       x_dims.data(),
                       y_ndim - 2,
                       y_dims.data(),
                       x_broadcast_dims.data(),
                       y_broadcast_dims.data(),
                       out_broadcast_dims.data());
  out_broadcast_dims[ndim - 2] = M;
  out_broadcast_dims[ndim - 1] = N;

  out->ResizeAndAllocate(common::make_ddim(out_broadcast_dims));
  dev_ctx.template Alloc<T>(out);

  const std::int64_t x_batch_size =
      std::accumulate(x_broadcast_dims.cbegin(),
                      x_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t y_batch_size =
      std::accumulate(y_broadcast_dims.cbegin(),
                      y_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t out_batch_size =
      std::accumulate(out_broadcast_dims.cbegin(),
                      out_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());

  // x_ndim >= 2 && y_ndim >= 2
  if (x_ndim == 2 && y_ndim == 2) {
    VLOG(3) << "Matmul's case 6";
    phi::MatMulGPUDNNKernelImpl<T>(dev_ctx, x, trans_x, y, trans_y, out);
    return;
  } else {
    // x_ndim == 2 && y_ndim >= 3 ||
    // y_ndim == 2 && x_ndim >= 3 ||
    // x_ndim >= 3 && y_ndim >= 3
    if (x_ndim == 2 && y_ndim >= 3) {
      // y might be folded, then call mm, but the result from muDNN needs to be
      // transposed, which introduce memcpy overhead, just use BMM in this case.
      VLOG(3) << "Matmul's case 7";

      std::vector<int> maybe_squeezed_y_dims = {
          static_cast<int>(y_batch_size),
          static_cast<int>(y_dims[ndim - 2]),
          static_cast<int>(y_dims[ndim - 1])};
      std::vector<int> maybe_squeezed_out_dims = {
          static_cast<int>(out_batch_size),
          static_cast<int>(out_broadcast_dims[ndim - 2]),
          static_cast<int>(out_broadcast_dims[ndim - 1])};
      phi::BmmGPUDNNKernelImpl<T>(dev_ctx,
                                  x,
                                  trans_x,
                                  common::vectorize<int>(x.dims()),
                                  y,
                                  trans_y,
                                  maybe_squeezed_y_dims,
                                  out,
                                  maybe_squeezed_out_dims);
      return;

    } else if (y_ndim == 2 && x_ndim >= 3) {
      // x might be folded, then call mm, which has a better performance than
      // BMM.
      VLOG(3) << "Matmul's case 8";
      auto x_tmp =
          trans_x ? phi::TransposeLast2Dim<T>(dev_ctx, x)
                  : paddle::experimental::CheckAndTrans2NewContiguousTensor(x);
      const std::vector<int> x_tmp_dims = common::vectorize<int>(x_tmp.dims());
      const std::int32_t folded_dim_x =
          std::accumulate(x_tmp_dims.cbegin(),
                          x_tmp_dims.cend() - 1,
                          static_cast<int32_t>(1),
                          std::multiplies<std::int32_t>());
      std::vector<int> xx_dims = {folded_dim_x, x_tmp_dims.back()};
      std::vector<int> yy_dims = common::vectorize<int>(y.dims());
      std::vector<int> out_tmp_dims = {
          folded_dim_x, static_cast<int>(out_broadcast_dims[ndim - 1])};
      phi::MatMulGPUDNNKernelImpl<T>(dev_ctx,
                                     x_tmp,
                                     false,
                                     xx_dims,
                                     y,
                                     trans_y,
                                     yy_dims,
                                     out,
                                     out_tmp_dims);
      return;
    } else {
      // x_ndim >= 3 && y_ndim >= 3
      // also take broadcast cases into consideration
      const bool is_broadcast_dims =
          !std::equal(x_broadcast_dims.cbegin(),
                      x_broadcast_dims.cbegin() + batch_dim,
                      y_broadcast_dims.cbegin());
      if (!is_broadcast_dims) {
        VLOG(3) << "Matmul's case 9";
        std::vector<int> maybe_squeezed_x_dims = {
            static_cast<int>(x_batch_size),
            static_cast<int>(x_dims[ndim - 2]),
            static_cast<int>(x_dims[ndim - 1])};
        std::vector<int> maybe_squeezed_y_dims = {
            static_cast<int>(y_batch_size),
            static_cast<int>(y_dims[ndim - 2]),
            static_cast<int>(y_dims[ndim - 1])};
        std::vector<int> maybe_squeezed_out_dims = {
            static_cast<int>(out_batch_size),
            static_cast<int>(out_broadcast_dims[ndim - 2]),
            static_cast<int>(out_broadcast_dims[ndim - 1])};
        phi::BmmGPUDNNKernelImpl<T>(dev_ctx,
                                    x,
                                    trans_x,
                                    maybe_squeezed_x_dims,
                                    y,
                                    trans_y,
                                    maybe_squeezed_y_dims,
                                    out,
                                    maybe_squeezed_out_dims);
        return;
      } else {
        // Compared to the MatMulFunctionImplWithBlas, the current
        // implementation performs slightly worse in large cases (512MB)
        VLOG(3) << "Matmul's case 10";

        const bool x_need_broadcast =
            !std::equal(x_broadcast_dims.cbegin(),
                        x_broadcast_dims.cbegin() + batch_dim,
                        out_broadcast_dims.cbegin());

        const bool y_need_broadcast =
            !std::equal(y_broadcast_dims.cbegin(),
                        y_broadcast_dims.cbegin() + batch_dim,
                        out_broadcast_dims.cbegin());
        DenseTensor x_helper;
        DenseTensor y_helper;
        if (x_need_broadcast) {
          std::vector<int> x_expand_shape(out_broadcast_dims.begin(),
                                          out_broadcast_dims.end());
          x_expand_shape[ndim - 2] = x_dims[x_ndim - 2];
          x_expand_shape[ndim - 1] = x_dims[x_ndim - 1];
          phi::ExpandKernel<T, Context>(dev_ctx, x, x_expand_shape, &x_helper);
          y_helper = y;
        }
        if (y_need_broadcast) {
          std::vector<int> y_expand_shape(out_broadcast_dims.begin(),
                                          out_broadcast_dims.end());
          y_expand_shape[ndim - 2] = y_dims[y_ndim - 2];
          y_expand_shape[ndim - 1] = y_dims[y_ndim - 1];
          phi::ExpandKernel<T, Context>(dev_ctx, y, y_expand_shape, &y_helper);
          x_helper = x;
        }
        std::vector<int> maybe_squeezed_x_dims = {
            static_cast<int>(out_batch_size),
            static_cast<int>(x_dims[x_ndim - 2]),
            static_cast<int>(x_dims[x_ndim - 1])};
        std::vector<int> maybe_squeezed_y_dims = {
            static_cast<int>(out_batch_size),
            static_cast<int>(y_dims[y_ndim - 2]),
            static_cast<int>(y_dims[y_ndim - 1])};
        std::vector<int> maybe_squeezed_out_dims = {
            static_cast<int>(out_batch_size),
            static_cast<int>(out_broadcast_dims[ndim - 2]),
            static_cast<int>(out_broadcast_dims[ndim - 1])};
        phi::BmmGPUDNNKernelImpl<T>(dev_ctx,
                                    x_helper,
                                    trans_x,
                                    maybe_squeezed_x_dims,
                                    y_helper,
                                    trans_y,
                                    maybe_squeezed_y_dims,
                                    out,
                                    maybe_squeezed_out_dims);
      }
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(matmul,  // musa_only
                   GPUDNN,
                   ALL_LAYOUT,
                   phi::MatmulGPUDNNKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
