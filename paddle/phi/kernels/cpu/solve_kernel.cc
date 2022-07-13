/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/solve_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/solve_kernel_impl.h"

namespace phi {

using Tensor = framework::Tensor;
using framework::To32BitIndex;

constexpr int kMULMKLDNNINT8 = 1;

template <typename DeviceContext, typename T>
void ReduceSumForSolve(const Tensor* input,
                       Tensor* output,
                       const std::vector<int>& reduce_dims,
                       bool keep_dim,
                       const paddle::framework::ExecutionContext& ctx) {
#if defined(__NVCC__) || defined(__HIPCC__)
  auto stream = ctx.cuda_device_context().stream();
  TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
      ctx.cuda_device_context(),
      *input,
      output,
      kps::IdentityFunctor<T>(),
      reduce_dims,
      stream);
#else
  ReduceKernelFunctor<DeviceContext, T, ops::SumFunctor>(
      input, output, reduce_dims, keep_dim, false, ctx)
      .template apply<T>();
#endif
}

// check the input other is vector_case or not
static inline bool is_vector_rhs(const DenseTensor& input,
                                 const DenseTensor& other) {
  auto x_dim = input.dims();
  auto y_dim = other.dims();
  auto x_dim_size = x_dim.size();
  auto y_dim_size = y_dim.size();
  std::vector<int64_t> x_dims_vec = phi::vectorize(x_dim);
  std::vector<int64_t> y_dims_vec = phi::vectorize(y_dim);

  std::vector<int64_t>::const_iterator f = x_dims_vec.begin();
  std::vector<int64_t>::const_iterator l = x_dims_vec.end() - 1;
  std::vector<int64_t> x_dims_vec_cut(f, l);  // input.shape[:-1]

  std::vector<int64_t> expected_batched_rhs_shape(x_dims_vec_cut);
  bool vector_case =
      y_dim_size == 1 || (x_dim_size - 1 == y_dim_size &&
                          y_dims_vec == (expected_batched_rhs_shape));

  return vector_case;
}

// unsqueeze operation helper
static framework::DDim GetOutputShapeUnsqueeze(
    const std::vector<int> unsqz_dims, const framework::DDim& in_dims) {
  int output_size = in_dims.size() + static_cast<int>(unsqz_dims.size());
  int cur_output_size = in_dims.size();
  std::vector<int64_t> output_shape(output_size, 0);

  // Validity Check: rank range.
  PADDLE_ENFORCE_LE(output_size,
                    6,
                    platform::errors::InvalidArgument(
                        "The output "
                        "tensor's rank should be less than 6."));

  for (int axis : unsqz_dims) {
    int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
    // Vaildity Check: the axis bound
    PADDLE_ENFORCE_GE(
        cur,
        0,
        platform::errors::InvalidArgument("The insert dimension value should "
                                          "not be less than 0"));
    PADDLE_ENFORCE_LE(cur,
                      cur_output_size,
                      platform::errors::InvalidArgument(
                          "The insert dimension value shoule not be larger "
                          "than the dimension size of input tensor"));
    // Move old axis, and insert new axis
    for (int i = cur_output_size; i >= cur; --i) {
      if (output_shape[i] == 1) {
        // Move axis
        output_shape[i + 1] = 1;
        output_shape[i] = 0;
      }
    }
    output_shape[cur] = 1;
    // Add the output size.
    cur_output_size++;
  }

  // Make output shape
  for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
    if (output_shape[out_idx] == 0) {
      output_shape[out_idx] = in_dims[in_idx++];
    }
  }

  return phi::make_ddim(output_shape);
}

// operation like squeeze(-1)
static void to_squeeze(const framework::ExecutionContext& context,
                       const DenseTensor& in,
                       DenseTensor* out) {
  auto x_dims = in.dims();
  std::vector<int> sqz_dims = {-1};
  auto out_dims = GetOutputShape(sqz_dims, x_dims, true);
  //   out->mutable_data(context.GetPlace(), in.type());
  dev_ctx.Alloc(out, in.type());
  framework::TensorCopy(
      in,
      context.GetPlace(),
      context.template device_context<platform::DeviceContext>(),
      out);
  out->Resize(out_dims);
}

// vector_case, need to operate like unsqueeze(-1)
static void to_unsqueeze(const framework::ExecutionContext& context,
                         const framework::Tensor& in,
                         framework::Tensor* out) {
  auto x_dims = in.dims();
  std::vector<int> unsqz_dims = {-1};
  framework::DDim out_dims = out->dims();
  out_dims = GetOutputShapeUnsqueeze(unsqz_dims, x_dims);
  framework::TensorCopy(
      in,
      context.GetPlace(),
      context.template device_context<platform::DeviceContext>(),
      out);
  out->Resize(out_dims);
}

// Prepared for the broadcast operation
static std::vector<int64_t> get_broadcast_batch_portion(
    std::vector<int64_t> x, std::vector<int64_t> y) {
  size_t size_x = x.size();
  size_t size_y = y.size();
  size_t size = std::max(size_x, size_y);
  std::vector<int64_t> batchPortion(size);

  ptrdiff_t i = (ptrdiff_t)size - 1;
  for (; i >= 0; --i) {
    ptrdiff_t offset = size - i - 1;
    ptrdiff_t dim_x = size_x - offset - 1;
    ptrdiff_t dim_y = size_y - offset - 1;
    int64_t x_size = (dim_x >= 0) ? x[dim_x] : 1;
    int64_t y_size = (dim_y >= 0) ? y[dim_y] : 1;

    PADDLE_ENFORCE_EQ(
        (x_size == y_size || x_size == 1 || y_size == 1),
        true,
        platform::errors::PreconditionNotMet(
            "The size of tensor x (%d) must match the size of tensor y "
            "(%d) at non-singleton dimension %d.",
            x_size,
            y_size,
            i));

    batchPortion[i] = x_size != 1 ? x_size : y_size;
  }
  return batchPortion;
}

// broadcast the batch dimensions of tensor x and tensor y.
static inline std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_broadcast_dims(const Tensor& x, const Tensor& y) {
  std::vector<int64_t> x_dims_vec = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims_vec = phi::vectorize(y.dims());

  std::vector<int64_t>::const_iterator f1 = x_dims_vec.begin();
  std::vector<int64_t>::const_iterator l1 = x_dims_vec.end() - 2;
  std::vector<int64_t> x_dims_vec_cut(f1, l1);

  std::vector<int64_t>::const_iterator f2 = y_dims_vec.begin();
  std::vector<int64_t>::const_iterator l2 = y_dims_vec.end() - 2;
  std::vector<int64_t> y_dims_vec_cut(f2, l2);

  std::vector<int64_t> expand_batch_portion =
      get_broadcast_batch_portion(x_dims_vec_cut, y_dims_vec_cut);

  std::vector<int64_t> x_expand_size({expand_batch_portion});
  x_expand_size.insert(x_expand_size.end(),
                       {x_dims_vec[static_cast<int>(x_dims_vec.size()) - 2],
                        x_dims_vec[static_cast<int>(x_dims_vec.size()) - 1]});

  std::vector<int64_t> y_expand_size({expand_batch_portion});
  y_expand_size.insert(y_expand_size.end(),
                       {y_dims_vec[static_cast<int>(y_dims_vec.size()) - 2],
                        y_dims_vec[static_cast<int>(y_dims_vec.size()) - 1]});

  return std::make_tuple(x_expand_size, y_expand_size);
}

template <int Rank, typename T, typename DeviceContext>
void expand_impl(const DeviceContext& context,
                 const Tensor& in,
                 Tensor* out,
                 const std::vector<int64_t>& expand_shape) {
  auto vec_in_dims = phi::vectorize<int>(in.dims());
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int> repeat_times(vec_in_dims.size());

  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    PADDLE_ENFORCE_NE(
        expand_shape[i],
        0,
        platform::errors::InvalidArgument("The expanded size cannot be zero."));
    if (i < diff) {
      PADDLE_ENFORCE_GT(
          expand_shape[i],
          0,
          platform::errors::InvalidArgument(
              "The expanded size (%d) for non-existing dimensions must be "
              "positive for expand operation.",
              expand_shape[i]));
      repeat_times[i] = expand_shape[i];
    } else if (expand_shape[i] > 0) {
      if (vec_in_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i],
            expand_shape[i],
            platform::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in shape for expand operation.",
                vec_in_dims[i],
                expand_shape[i]));
        repeat_times[i] = 1;
      } else {
        repeat_times[i] = expand_shape[i];
      }
    } else {
      PADDLE_ENFORCE_EQ(
          expand_shape[i],
          -1,
          platform::errors::InvalidArgument(
              "When the value in shape is negative for expand_v2 op, "
              "only -1 is supported, but the value received is %d.",
              expand_shape[i]));
      repeat_times[i] = 1;
    }
  }

  Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    bcast_dims[i] = repeat_times[i];
  }

  framework::DDim new_in_dims = phi::make_ddim(vec_in_dims);
  framework::DDim out_dims(new_in_dims);
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    out_dims[i] *= repeat_times[i];
  }

  out->Resize(out_dims);
  out->mutable_data<T>(context.GetPlace());
  auto x = EigenTensor<T, Rank>::From(in, new_in_dims);
  auto y = EigenTensor<T, Rank>::From(*out, out_dims);
  auto& place = *context.eigen_device();
  // use 32-bit index to speed up
  bool use_32bit_index = y.size() < Eigen::NumTraits<int>::highest();
  if (use_32bit_index) {
    EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, To32BitIndex(y), To32BitIndex(x), bcast_dims);
  } else {
    EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, y, x, bcast_dims);
  }
}

template <typename T, typename DeviceContext>
void TensorExpand(const DeviceContext& context,
                  const Tensor& in,
                  Tensor* out,
                  const std::vector<int64_t>& expand_shape) {
  // necessary check before expand operation
  PADDLE_ENFORCE_GE(expand_shape.size(),
                    in.dims().size(),
                    platform::errors::InvalidArgument(
                        "The size of 'expand_shape' (%d) should >= the input "
                        "Tensor's rank (%d).",
                        expand_shape.size(),
                        in.dims().size()));
  PADDLE_ENFORCE_LE(expand_shape.size(),
                    MAX_RANK_SUPPORTED,
                    platform::errors::InvalidArgument(
                        "The size of 'expand_shape' (%d) should be <= %d",
                        expand_shape.size(),
                        MAX_RANK_SUPPORTED));
  switch (expand_shape.size()) {
    case 1:
      expand_impl<1, T, DeviceContext>(context, in, out, expand_shape);
      break;
    case 2:
      expand_impl<2, T, DeviceContext>(context, in, out, expand_shape);
      break;
    case 3:
      expand_impl<3, T, DeviceContext>(context, in, out, expand_shape);
      break;
    case 4:
      expand_impl<4, T, DeviceContext>(context, in, out, expand_shape);
      break;
    case 5:
      expand_impl<5, T, DeviceContext>(context, in, out, expand_shape);
      break;
    case 6:
      expand_impl<6, T, DeviceContext>(context, in, out, expand_shape);
      break;
  }
}

template <typename Context, typename T>
static void linalg_solve(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         DenseTensor* out) {
  //   out->mutable_data<T>(context.GetPlace());
  auto out = dev_ctx.template Alloc<T>(out);

  //   auto& dev_ctx = context.template device_context<Context>();
  math::MatrixSolveFunctor<Context, T> mat_solve;

  // input y can be vector or matrix
  // but need to be unsqueezed if y is a vector
  bool is_vector = false;
  is_vector = is_vector_rhs(*x, *y);

  Tensor tmp_y;
  if (is_vector) {
    tmp_y.mutable_data(context.GetPlace(), y->dtype());
    to_unsqueeze(context, *y, &tmp_y);
  } else {
    tmp_y.Resize(y->dims());
    tmp_y.mutable_data(context.GetPlace(), y->dtype());
    framework::TensorCopy(
        *y,
        context.GetPlace(),
        context.template device_context<platform::DeviceContext>(),
        &tmp_y);
  }

  Tensor tmp_x;
  tmp_x.Resize(x->dims());
  tmp_x.mutable_data(context.GetPlace(), x->dtype());
  framework::TensorCopy(
      *x,
      context.GetPlace(),
      context.template device_context<platform::DeviceContext>(),
      &tmp_x);

  std::vector<int64_t> x_broadcast_dims;
  std::vector<int64_t> y_broadcast_dims;
  std::tie(x_broadcast_dims, y_broadcast_dims) =
      get_broadcast_dims(tmp_x, tmp_y);

  Tensor tmp_x_bc;
  TensorExpand<T, DeviceContext>(dev_ctx, tmp_x, &tmp_x_bc, x_broadcast_dims);

  Tensor tmp_y_bc;
  TensorExpand<T, DeviceContext>(dev_ctx, tmp_y, &tmp_y_bc, y_broadcast_dims);

  auto x_dim = x->dims();
  auto y_dim = y->dims();
  auto x_dim_size = x_dim.size();
  auto y_dim_size = y_dim.size();

  if (is_vector) {                 // vector case
    out->Resize(tmp_y_bc.dims());  // out.unsqueeze(-1)
    mat_solve(dev_ctx, tmp_x_bc, tmp_y_bc, out);

    Tensor out_tmp;
    out_tmp.Resize(out->dims());
    out_tmp = *out;
    to_squeeze(context, out_tmp, out);  // out.squeeze(-1)
  } else {
    PADDLE_ENFORCE_EQ(
        x_dim[x_dim_size - 1],
        y_dim[y_dim_size - 2],
        platform::errors::InvalidArgument(
            "Matrix X1 with dimension greater than 2 and any matrix Y1,"
            "the matrix X1's width must be equal with matrix Y1's "
            "height. But received X's shape = [%s], X1's shape = [%s], X1's "
            "width = %s; Y's shape = [%s], Y1's shape = [%s], Y1's height = "
            "%s.",
            x_dim,
            x_dim,
            x_dim[x_dim_size - 1],
            y_dim,
            y_dim,
            y_dim[y_dim_size - 2]));
    mat_solve(dev_ctx, tmp_x_bc, tmp_y_bc, out);
  }
}

template <typename T, typename Context>
void SolveKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 DenseTensor* out) {
  linalg_solve<Context, T>(dev_ctx, x, y, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(solve, CPU, ALL_LAYOUT, phi::SolveKernel, float, double) {}
