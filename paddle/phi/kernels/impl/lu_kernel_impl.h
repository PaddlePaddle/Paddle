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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "paddle/phi/kernels/funcs/tril_triu_compute.h"
#include "paddle/phi/kernels/impl/set_value_kernel_impl.h"

namespace phi {

template <typename T>
using SubFunctor = phi::funcs::SubtractFunctor<T>;

template <typename Context, typename T, size_t D>
void SetValueCompute(const Context& dev_ctx,
                     DenseTensor* in,
                     DenseTensor* value_tensor,
                     DenseTensor* out,
                     const std::vector<int64_t>& axes,
                     std::vector<int64_t>* starts,
                     std::vector<int64_t>* ends,
                     const std::vector<int64_t>& shape) {
  std::vector<int64_t> steps = {1, 1};
  std::vector<int64_t> decrease_axes = {};
  std::vector<int64_t> none_axes = {};

  auto dtype = in->dtype();

  auto in_dims = in->dims();
  phi::funcs::CheckAndUpdateSliceAttrs<int64_t>(
      in_dims, axes, starts, ends, &steps);
  auto slice_dims =
      phi::funcs::GetSliceDims(in_dims, axes, *starts, *ends, &steps);
  auto decrease_slice_dims =
      phi::funcs::GetDecreasedDims(slice_dims, decrease_axes);

  auto slice_dims_for_assign = decrease_slice_dims;
  if (!none_axes.empty()) {
    std::vector<int64_t> slice_dims_with_none;

    size_t none_axes_cur = 0, decrease_axes_cur = 0;
    for (int i = 0; i < slice_dims.size(); ++i) {
      while (none_axes_cur < none_axes.size() &&
             none_axes[none_axes_cur] <= i) {
        slice_dims_with_none.push_back(1);
        none_axes_cur++;
      }
      if (decrease_axes_cur < decrease_axes.size() &&
          decrease_axes[decrease_axes_cur] == i) {
        decrease_axes_cur++;
      } else {
        slice_dims_with_none.push_back(slice_dims[i]);
      }
    }
    while (none_axes_cur < none_axes.size()) {
      slice_dims_with_none.push_back(1);
      none_axes_cur++;
    }

    slice_dims_for_assign = common::make_ddim(slice_dims_with_none);
  }

  auto place = dev_ctx.GetPlace();
  auto& eigen_place = *dev_ctx.eigen_device();

  // Here copy data from input to avoid data loss at PE and Graph level.
  // TODO(liym27): Speed up in the future version.
  // - Q: Why don't call ShareDataWith to speed up?
  // - A: Because it's not supported to ShareDataWith on OP's input and output
  // https://github.com/PaddlePaddle/Paddle/wiki/ShareDataWith-and-ShareBufferWith-are-prohibited-in-OP
  // - Q: Why don't delete Input, after all, the input and output are the same
  // Tensor at program level?
  // - A: If deleting Input, the graph will be complex, such as there will
  // be two ops points to the output in graph: op1 -> output <- set_value.
  // In this case, we have to find a way to handle the running order of
  // set_value is what we want.
  phi::Copy(dev_ctx, *in, place, false, out);

  DenseTensor slice_tensor(dtype), pad_tensor(dtype);
  slice_tensor.Resize(slice_dims);
  dev_ctx.template Alloc<T>(&slice_tensor);
  pad_tensor.Resize(in_dims);
  dev_ctx.template Alloc<T>(&pad_tensor);

  auto pad_e = EigenTensor<T, D>::From(pad_tensor, in_dims);
  auto out_e = EigenTensor<T, D>::From(*out);
  auto slice_e = EigenTensor<T, D>::From(slice_tensor, slice_dims);

  // Step 1: Set the value of out at `_index` to zero
  slice_e.device(eigen_place) = slice_e.constant(T(0));

  auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

  for (size_t i = 0; i < D; ++i) {
    starts_indices[i] = 0;
    ends_indices[i] = slice_dims[i];
    strides_indices[i] = 1;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int axis_index = axes[i];
    starts_indices[axis_index] = (*starts)[i];
    ends_indices[axis_index] = (*ends)[i];
    strides_indices[axis_index] = steps[i];
    if ((*starts)[i] ==
        (*ends)[i]) {  // slice is empty, data will not be changed
      return;
    }
  }

  out_e.stridedSlice(starts_indices, ends_indices, strides_indices)
      .device(eigen_place) = slice_e;

  // Step 2: Set a tensor with the same shape as out tensor. And its data at
  // '_index' is the same as value_tensor, and data out of '_index' to zero

  // - Step 2.1 Set slice tensor with value

  // NOTE(liym27): [ Why resize slice_tensor here? ]
  // A: When do broadcasting on slice_tensor and value_tensor, the shape of
  // slice_tensor should be decreased dims.
  // e.g.
  //  x[:,0] = value_tensor
  // x's shape = [3, 4], value_tensor's shape = [3]
  // We get slice_dims = [3, 1],  decrease_slice_dims = [3]
  // If do broadcasting on Tensor with shape [3, 1] and [3], the result's
  // shape is [3, 3], which cross the border;
  // If do broadcasting on Tensor with shape [3] and [3], the result's shape
  // is [3], which is right.

  slice_tensor.Resize(slice_dims_for_assign);
  if (value_tensor != nullptr) {
    CheckIsDimsMatch(slice_dims_for_assign, value_tensor->dims());
    phi::funcs::ElementwiseCompute<SubFunctor<T>, T>(
        dev_ctx, slice_tensor, *value_tensor, SubFunctor<T>(), &slice_tensor);
  } else {
    DenseTensor value_t(dtype);
    auto value_dims = common::make_ddim(shape);
    CheckIsDimsMatch(slice_dims_for_assign, value_dims);

    value_t.Resize(value_dims);
    dev_ctx.template Alloc<T>(&value_t);
    phi::funcs::ElementwiseCompute<SubFunctor<T>, T>(
        dev_ctx, slice_tensor, value_t, SubFunctor<T>(), &slice_tensor);
  }
  slice_tensor.Resize(slice_dims);

  // - Step 2.2 Pad slice tensor with 0
  pad_e.device(eigen_place) = pad_e.constant(T(0));
  pad_e.stridedSlice(starts_indices, ends_indices, strides_indices)
      .device(eigen_place) = slice_e;

  // Step 3: Set out tensor with value_tensor
  out_e.device(eigen_place) = out_e - pad_e;
}

template <typename Context, typename T>
void SetValueCompute_dispatch(const Context& dev_ctx,
                              DenseTensor* in,
                              DenseTensor* value_tensor,
                              DenseTensor* out,
                              const std::vector<int64_t>& axes,
                              std::vector<int64_t>* starts,
                              std::vector<int64_t>* ends,
                              const std::vector<int64_t>& shape,
                              int rank) {
  switch (rank) {
    case 1:
      SetValueCompute<Context, T, 1>(
          dev_ctx, in, value_tensor, out, axes, starts, ends, shape);
      break;
    case 2:
      SetValueCompute<Context, T, 2>(
          dev_ctx, in, value_tensor, out, axes, starts, ends, shape);
      break;
    case 3:
      SetValueCompute<Context, T, 3>(
          dev_ctx, in, value_tensor, out, axes, starts, ends, shape);
      break;
    case 4:
      SetValueCompute<Context, T, 4>(
          dev_ctx, in, value_tensor, out, axes, starts, ends, shape);
      break;
    case 5:
      SetValueCompute<Context, T, 5>(
          dev_ctx, in, value_tensor, out, axes, starts, ends, shape);
      break;
    case 6:
      SetValueCompute<Context, T, 6>(
          dev_ctx, in, value_tensor, out, axes, starts, ends, shape);
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.", rank));
  }
}

template <typename Context, typename T>
void Tensor_Conj(const Context& dev_ctx,
                 const DenseTensor& tensor,
                 DenseTensor* out) {
  out->Resize(tensor.dims());
  phi::funcs::ForRange<Context> out_for_range(dev_ctx, tensor.numel());
  dev_ctx.template Alloc<T>(out);
  phi::funcs::ConjFunctor<T> out_functor(
      tensor.data<T>(), tensor.numel(), out->data<T>());
  out_for_range(out_functor);
}

template <typename Context, typename T>
void Tensor_Add(const Context& dev_ctx,
                const DenseTensor& src1,
                const DenseTensor& src2,
                DenseTensor* out) {
  out->Resize(src1.dims());
  dev_ctx.template Alloc<T>(out);

  phi::AddKernel<T, Context>(dev_ctx, src1, src2, out);
}

template <typename Context, typename T>
void Tensor_Sub(const Context& dev_ctx,
                const DenseTensor& src1,
                const DenseTensor& src2,
                DenseTensor* out) {
  out->Resize(src1.dims());
  dev_ctx.template Alloc<T>(out);

  phi::SubtractKernel<T, Context>(dev_ctx, src1, src2, out);
}

template <typename Context, typename T, size_t D>
void SliceCompute(const Context& dev_ctx,
                  const DenseTensor* in,
                  DenseTensor* out,
                  const std::vector<int>& axes_int,
                  const std::vector<int>& starts_int,
                  const std::vector<int>& ends_int) {
  std::vector<int64_t> axes(axes_int.begin(), axes_int.end());
  std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
  std::vector<int64_t> ends(ends_int.begin(), ends_int.end());

  std::vector<int> decrease_axis = {};
  std::vector<int> infer_flags = {};

  PADDLE_ENFORCE_EQ(
      starts.size(),
      axes.size(),
      common::errors::InvalidArgument(
          "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends.size(),
                    axes.size(),
                    common::errors::InvalidArgument(
                        "The size of ends must be equal to the size of axes."));

  // Step 2: Compute output

  auto in_dims = in->dims();
  auto out_dims = out->dims();
  auto slice_dims = out_dims;

  // 2.1 Infer output dims
  for (size_t i = 0; i < axes.size(); ++i) {
    // when start == -1 && end == start+1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[axes[i]];
      }
    }
  }

  phi::funcs::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
  slice_dims = phi::funcs::GetSliceDims<int64_t>(
      in_dims, axes, starts, ends, nullptr, nullptr);
  out_dims = phi::funcs::GetDecreasedDims(slice_dims, decrease_axis);

  // 2.2 Get output
  auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();

  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = slice_dims[i];
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    offsets[axes[i]] = starts[i];
  }

  out->Resize(slice_dims);
  dev_ctx.template Alloc<T>(out);

  auto in_t = EigenTensor<T, D>::From(*in, in_dims);
  auto out_t = EigenTensor<T, D>::From(*out, slice_dims);
  auto& eigen_place = *dev_ctx.eigen_device();

  if (in->numel() <= Eigen::NumTraits<int>::highest()) {
    // similar to tf.slice:
    // if element number less than INT_MAX, change the type of index to int
    Eigen::DSizes<int, D> offsets_32bit, extents_32bit;
    for (size_t i = 0; i < D; i++) {
      offsets_32bit[i] = offsets[i];
      extents_32bit[i] = extents[i];
    }
    funcs::EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
        eigen_place,
        To32BitIndex(out_t),
        To32BitIndex(in_t),
        offsets_32bit,
        extents_32bit);
  } else {
    funcs::EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
        eigen_place, out_t, in_t, offsets, extents);
  }

  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
}

template <typename Context, typename T>
void Tensor_narrow(const Context& dev_ctx,
                   const DenseTensor* src,
                   DenseTensor* out,
                   int row_s,
                   int row_e,
                   int col_s,
                   int col_e) {
  auto rank = src->dims().size();
  std::vector<int> axes_int = {rank - 2, rank - 1};
  std::vector<int> starts_int = {row_s, col_s};
  std::vector<int> ends_int = {row_e, col_e};
  switch (rank) {
    case 1:
      SliceCompute<Context, T, 1>(
          dev_ctx, src, out, axes_int, starts_int, ends_int);
      break;
    case 2:
      SliceCompute<Context, T, 2>(
          dev_ctx, src, out, axes_int, starts_int, ends_int);
      break;
    case 3:
      SliceCompute<Context, T, 3>(
          dev_ctx, src, out, axes_int, starts_int, ends_int);
      break;
    case 4:
      SliceCompute<Context, T, 4>(
          dev_ctx, src, out, axes_int, starts_int, ends_int);
      break;
    case 5:
      SliceCompute<Context, T, 5>(
          dev_ctx, src, out, axes_int, starts_int, ends_int);
      break;
    case 6:
      SliceCompute<Context, T, 6>(
          dev_ctx, src, out, axes_int, starts_int, ends_int);
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.", rank));
  }
}

template <typename Context>
void arange(const Context& dev_ctx,
            DenseTensor* tmp,
            int w,
            int batchsize = 1,
            int h = 1) {
  tmp->Resize(common::make_ddim({batchsize * w}));
  dev_ctx.template HostAlloc<int32_t>(tmp);
  auto tmpdata = tmp->data<int32_t>();
  for (int b = 0; b < batchsize; b++) {
    for (int i = 0; i < w; i++) {
      tmpdata[b * w + i] = static_cast<int32_t>(b * h + i);
    }
  }
}

template <typename T>
struct OneFunctor {
  OneFunctor(T* output, int* idtptr, int w, int dim)
      : output_(output), idtptr_(idtptr), w_(w), dim_(dim) {}

  HOSTDEVICE void operator()(size_t idx) const {
    output_[w_ * idtptr_[idx] + idx % dim_] = static_cast<T>(1);
  }

  T* output_;
  int* idtptr_;
  int w_;
  int dim_;
};

template <typename Context, typename T>
void LU_Unpack(const Context& dev_ctx,
               const DenseTensor* LU,
               DenseTensor* L,
               DenseTensor* U) {
  const auto udims = LU->dims();
  L->Resize(udims);
  U->Resize(udims);
  const auto H = udims[udims.size() - 2];
  const auto W = udims[udims.size() - 1];
  dev_ctx.template Alloc<T>(L);
  auto L_dataptr = L->data<T>();
  phi::funcs::ForRange<Context> x_for_range(dev_ctx, LU->numel());
  phi::funcs::TrilTriuCompute<T> tril_computer(
      LU->data<T>(), -1, true, H, W, L_dataptr);
  x_for_range(tril_computer);

  dev_ctx.template Alloc<T>(U);
  phi::funcs::TrilTriuCompute<T> triu_computer(
      LU->data<T>(), 0, false, H, W, U->data<T>());
  x_for_range(triu_computer);

  // set L's diagonal 1
  auto dim = std::min(H, W);
  DenseTensor rowtensor, rt_dev;
  auto batchsize = product(common::slice_ddim(udims, 0, udims.size() - 2));

  // if udims is [0, ..., H, W], it should be 0
  if (udims.size() == 2) batchsize = std::max(static_cast<int>(batchsize), 1);

  arange<Context>(dev_ctx, &rowtensor, dim, batchsize, H);
  auto idtptr = rowtensor.data<int32_t>();
  if (phi::AllocationType::GPU == dev_ctx.GetPlace().GetType()) {
    phi::Copy(dev_ctx, rowtensor, dev_ctx.GetPlace(), false, &rt_dev);
    idtptr = rt_dev.data<int32_t>();
  }

  phi::funcs::ForRange<Context> for_range(dev_ctx, rowtensor.numel());
  OneFunctor<T> functor(L_dataptr, idtptr, W, dim);
  for_range(functor);
}

template <typename Context, typename T>
void scatterpivot(
    const Context& dev_ctx, T* out_data, DenseTensor* idlst, int w, int dim) {
  DenseTensor idlst_tmp;
  idlst_tmp.Resize(idlst->dims());
  dev_ctx.template Alloc<int32_t>(&idlst_tmp);
  phi::Copy(dev_ctx, *idlst, dev_ctx.GetPlace(), false, &idlst_tmp);
  auto idtptr = idlst_tmp.data<int32_t>();

  phi::funcs::ForRange<Context> for_range(dev_ctx, idlst_tmp.numel());
  OneFunctor<T> functor(out_data, idtptr, w, dim);
  for_range(functor);
}

template <typename Context, typename T>
void Unpack_Pivot(const Context& dev_ctx,
                  const DenseTensor& Pivot,
                  DenseTensor* P,
                  int h,
                  int w UNUSED) {
  auto dims = Pivot.dims();
  auto Pdimvec = common::vectorize(dims);
  auto prank = Pdimvec.size();
  auto Pnum = dims[prank - 1];
  DenseTensor Pivot_cpu;
  phi::CPUPlace cpu;
  phi::Copy(dev_ctx, Pivot, cpu, false, &Pivot_cpu);
  auto pdataptr = Pivot_cpu.data<int32_t>();
  Pdimvec[prank - 1] = h;
  Pdimvec.emplace_back(h);
  auto Pdim = common::make_ddim(Pdimvec);
  P->Resize(Pdim);
  dev_ctx.template Alloc<T>(P);
  auto pdata = P->data<T>();
  phi::funcs::SetConstant<Context, T> setter;
  setter(dev_ctx, P, static_cast<T>(0));

  auto batchsize = product(common::slice_ddim(dims, 0, prank - 1));
  if (prank == 1) batchsize = std::max(static_cast<int>(batchsize), 1);

  DenseTensor idt;
  for (int i = 0; i < batchsize; i++) {
    arange<Context>(dev_ctx, &idt, h);
    auto idlst = idt.data<int32_t>();
    for (int j = 0; j < Pnum; j++) {
      PADDLE_ENFORCE_EQ(
          (pdataptr[i * Pnum + j] > 0) && (pdataptr[i * Pnum + j] <= h),
          true,
          common::errors::InvalidArgument(
              "The data in Pivot must be between (1, x.shape[-2]],"
              "but got %d in Pivot while the x.shape[-2] is %d."
              "Please make sure that the inputs(x and Pivot) is the output of "
              "paddle.linalg.lu.",
              pdataptr[i * Pnum + j],
              h));
      if (idlst[pdataptr[i * Pnum + j] - 1] == idlst[j]) continue;
      auto temp = idlst[j];
      idlst[j] = idlst[pdataptr[i * Pnum + j] - 1];
      idlst[pdataptr[i * Pnum + j] - 1] = temp;
    }
    scatterpivot(dev_ctx, &(pdata[i * h * h]), &idt, h, h);
  }
}

template <typename Context, typename T>
DenseTensor Transpose2DTo6D(const Context& dev_ctx, const DenseTensor& x) {
  // transpose the last two dimision
  DenseTensor ret;
  auto x_dim = x.dims();
  auto x_vec = common::vectorize<int>(x_dim);
  int rank = x_vec.size();

  for (int i = 0; i < x_dim.size(); i++) {
    PADDLE_ENFORCE_LT(0,
                      x_dim[i],
                      errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));
  }

  std::swap(x_vec[rank - 1], x_vec[rank - 2]);
  std::vector<int> out_shape = x_vec;
  std::vector<int> axis(rank);
  for (int i = 0; i < rank; ++i) {
    axis[i] = i;
  }
  std::swap(axis[rank - 1], axis[rank - 2]);
  ret.Resize(common::make_ddim(x_vec));
  dev_ctx.template Alloc<T>(&ret);
  switch (rank) {
    case 2: {
      phi::funcs::Transpose<Context, T, 2> trans;
      trans(dev_ctx, x, &ret, axis);
      break;
    }
    case 3: {
      phi::funcs::Transpose<Context, T, 3> trans;
      trans(dev_ctx, x, &ret, axis);
      break;
    }
    case 4: {
      phi::funcs::Transpose<Context, T, 4> trans;
      trans(dev_ctx, x, &ret, axis);
      break;
    }
    case 5: {
      phi::funcs::Transpose<Context, T, 5> trans;
      trans(dev_ctx, x, &ret, axis);
      break;
    }
    case 6: {
      phi::funcs::Transpose<Context, T, 6> trans;
      trans(dev_ctx, x, &ret, axis);
      break;
    }
    default: {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Invalid Rank number, "
          "currently only support rank between 2~6"));
    }
  }
  return ret;
}

}  // namespace phi
