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

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/operators/set_value_op.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/funcs/tril_triu_compute.h"
#include "paddle/phi/kernels/triangular_solve_kernel.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensorArray = framework::LoDTensorArray;

template <typename DeviceContext, typename T, size_t D>
void SetValueCompute(const framework::ExecutionContext& ctx,
                     framework::Tensor* in, framework::Tensor* value_tensor,
                     framework::Tensor* out, const std::vector<int64_t>& axes,
                     std::vector<int64_t>* starts, std::vector<int64_t>* ends,
                     const std::vector<int64_t>& shape) {
  std::vector<int64_t> steps = {1, 1};
  std::vector<int64_t> decrease_axes = {};
  std::vector<int64_t> none_axes = {};

  auto dtype = framework::TransToProtoVarType(in->dtype());

  auto in_dims = in->dims();
  CheckAndUpdateSliceAttrs<int64_t>(in_dims, axes, starts, ends, &steps);
  auto slice_dims = GetSliceDims(in_dims, axes, *starts, *ends, &steps);
  auto decrease_slice_dims = GetDecreasedDims(slice_dims, decrease_axes);

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

    slice_dims_for_assign = phi::make_ddim(slice_dims_with_none);
  }

  auto place = ctx.GetPlace();
  auto& eigen_place =
      *ctx.template device_context<DeviceContext>().eigen_device();

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
  paddle::framework::TensorCopy(*in, place, out);

  Tensor slice_tensor(framework::TransToPhiDataType(dtype)),
      pad_tensor(framework::TransToPhiDataType(dtype));
  slice_tensor.mutable_data<T>(slice_dims, place);
  pad_tensor.mutable_data<T>(in_dims, place);

  auto pad_e = framework::EigenTensor<T, D>::From(pad_tensor, in_dims);
  auto out_e = framework::EigenTensor<T, D>::From(*out);
  auto slice_e = framework::EigenTensor<T, D>::From(slice_tensor, slice_dims);

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
    // ElementwiseComputeEx can do broadcasting
    ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
        ctx, &slice_tensor, value_tensor, -1, SubFunctor<T>(), &slice_tensor);
  } else {
    Tensor value_t(framework::TransToPhiDataType(dtype));
    auto value_dims = phi::make_ddim(shape);
    CheckIsDimsMatch(slice_dims_for_assign, value_dims);

    value_t.mutable_data<T>(value_dims, place);
    auto value_name = GetValueName(dtype);
    CopyVecotorToTensor<T>(value_name.c_str(), &value_t, ctx);
    value_t.Resize(value_dims);
    ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
        ctx, &slice_tensor, &value_t, -1, SubFunctor<T>(), &slice_tensor);
  }
  slice_tensor.Resize(slice_dims);

  // - Step 2.2 Pad slice tensor with 0
  pad_e.device(eigen_place) = pad_e.constant(T(0));
  pad_e.stridedSlice(starts_indices, ends_indices, strides_indices)
      .device(eigen_place) = slice_e;

  // Step 3: Set out tensor with value_tensor
  out_e.device(eigen_place) = out_e - pad_e;
}

template <typename DeviceContext, typename T>
void SetValueCompute_dispatch(
    const framework::ExecutionContext& ctx, framework::Tensor* in,
    framework::Tensor* value_tensor, framework::Tensor* out,
    const std::vector<int64_t>& axes, std::vector<int64_t>* starts,
    std::vector<int64_t>* ends, const std::vector<int64_t>& shape, int rank) {
  switch (rank) {
    case 1:
      SetValueCompute<DeviceContext, T, 1>(ctx, in, value_tensor, out, axes,
                                           starts, ends, shape);
      break;
    case 2:
      SetValueCompute<DeviceContext, T, 2>(ctx, in, value_tensor, out, axes,
                                           starts, ends, shape);
      break;
    case 3:
      SetValueCompute<DeviceContext, T, 3>(ctx, in, value_tensor, out, axes,
                                           starts, ends, shape);
      break;
    case 4:
      SetValueCompute<DeviceContext, T, 4>(ctx, in, value_tensor, out, axes,
                                           starts, ends, shape);
      break;
    case 5:
      SetValueCompute<DeviceContext, T, 5>(ctx, in, value_tensor, out, axes,
                                           starts, ends, shape);
      break;
    case 6:
      SetValueCompute<DeviceContext, T, 6>(ctx, in, value_tensor, out, axes,
                                           starts, ends, shape);
      break;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.", rank));
  }
}

template <typename DeviceContext, typename T>
void Tensor_Conj(const DeviceContext& dev_ctx, const framework::Tensor& tensor,
                 framework::Tensor* out) {
  out->Resize(tensor.dims());
  platform::ForRange<DeviceContext> out_for_range(dev_ctx, tensor.numel());
  phi::funcs::ConjFunctor<T> out_functor(
      tensor.data<T>(), tensor.numel(),
      out->mutable_data<T>(dev_ctx.GetPlace()));
  out_for_range(out_functor);
}

template <typename DeviceContext, typename T>
void Tensor_Add(const DeviceContext& dev_ctx, const framework::Tensor& src1,
                const framework::Tensor& src2, framework::Tensor* out) {
  out->Resize(src1.dims());
  out->mutable_data<T>(dev_ctx.GetPlace());

  phi::AddRawKernel<
      T, typename paddle::framework::ConvertToPhiContext<DeviceContext>::TYPE>(
      static_cast<const typename paddle::framework::ConvertToPhiContext<
          DeviceContext>::TYPE&>(dev_ctx),
      src1, src2, -1, out);
}

template <typename DeviceContext, typename T>
void Tensor_Sub(const DeviceContext& dev_ctx, const framework::Tensor& src1,
                const framework::Tensor& src2, framework::Tensor* out) {
  out->Resize(src1.dims());
  out->mutable_data<T>(dev_ctx.GetPlace());

  phi::SubtractRawKernel<
      T, typename paddle::framework::ConvertToPhiContext<DeviceContext>::TYPE>(
      static_cast<const typename paddle::framework::ConvertToPhiContext<
          DeviceContext>::TYPE&>(dev_ctx),
      src1, src2, -1, out);
}

template <typename DeviceContext, typename T, size_t D>
void SliceCompute(const framework::ExecutionContext& ctx,
                  const framework::Tensor* in, framework::Tensor* out,
                  const std::vector<int>& axes_int,
                  const std::vector<int>& starts_int,
                  const std::vector<int>& ends_int) {
  std::vector<int64_t> axes(axes_int.begin(), axes_int.end());
  std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
  std::vector<int64_t> ends(ends_int.begin(), ends_int.end());

  std::vector<int> decrease_axis = {};
  std::vector<int> infer_flags = {};

  PADDLE_ENFORCE_EQ(
      starts.size(), axes.size(),
      platform::errors::InvalidArgument(
          "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends.size(), axes.size(),
                    platform::errors::InvalidArgument(
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

  CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
  slice_dims =
      GetSliceDims<int64_t>(in_dims, axes, starts, ends, nullptr, nullptr);
  out_dims = GetDecreasedDims(slice_dims, decrease_axis);

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
  out->mutable_data<T>(ctx.GetPlace());

  auto in_t = framework::EigenTensor<T, D>::From(*in, in_dims);
  auto out_t = framework::EigenTensor<T, D>::From(*out, slice_dims);
  auto& eigen_place =
      *ctx.template device_context<DeviceContext>().eigen_device();

  if (in->numel() <= Eigen::NumTraits<int>::highest()) {
    // similar to tf.slice:
    // if element number less than INT_MAX, change the type of index to int
    Eigen::DSizes<int, D> offsets_32bit, extents_32bit;
    for (size_t i = 0; i < D; i++) {
      offsets_32bit[i] = offsets[i];
      extents_32bit[i] = extents[i];
    }
    EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
        eigen_place, framework::To32BitIndex(out_t),
        framework::To32BitIndex(in_t), offsets_32bit, extents_32bit);
  } else {
    EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
        eigen_place, out_t, in_t, offsets, extents);
  }

  out->Resize(out_dims);
  out->mutable_data<T>(ctx.GetPlace());
}

template <typename DeviceContext, typename T>
void Tensor_narrow(const framework::ExecutionContext& ctx,
                   const framework::Tensor* src, framework::Tensor* out,
                   int row_s, int row_e, int col_s, int col_e) {
  auto rank = src->dims().size();
  std::vector<int> axes_int = {rank - 2, rank - 1};
  std::vector<int> starts_int = {row_s, col_s};
  std::vector<int> ends_int = {row_e, col_e};
  switch (rank) {
    case 1:
      SliceCompute<DeviceContext, T, 1>(ctx, src, out, axes_int, starts_int,
                                        ends_int);
      break;
    case 2:
      SliceCompute<DeviceContext, T, 2>(ctx, src, out, axes_int, starts_int,
                                        ends_int);
      break;
    case 3:
      SliceCompute<DeviceContext, T, 3>(ctx, src, out, axes_int, starts_int,
                                        ends_int);
      break;
    case 4:
      SliceCompute<DeviceContext, T, 4>(ctx, src, out, axes_int, starts_int,
                                        ends_int);
      break;
    case 5:
      SliceCompute<DeviceContext, T, 5>(ctx, src, out, axes_int, starts_int,
                                        ends_int);
      break;
    case 6:
      SliceCompute<DeviceContext, T, 6>(ctx, src, out, axes_int, starts_int,
                                        ends_int);
      break;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.", rank));
  }
}

template <typename DeviceContext>
void arange(const DeviceContext& dev_ctx, framework::Tensor* tmp, int w,
            int batchsize = 1, int h = 1) {
  tmp->Resize(phi::make_ddim({batchsize * w}));
  platform::CPUPlace cpu;
  auto tmpdata = tmp->mutable_data<int32_t>(cpu);
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

template <typename DeviceContext, typename T>
void LU_Unpack(const DeviceContext& dev_ctx, const framework::Tensor* LU,
               framework::Tensor* L, framework::Tensor* U) {
  const auto udims = LU->dims();
  L->Resize(udims);
  U->Resize(udims);
  const auto H = udims[udims.size() - 2];
  const auto W = udims[udims.size() - 1];
  auto L_dataptr = L->mutable_data<T>(dev_ctx.GetPlace());
  platform::ForRange<DeviceContext> x_for_range(dev_ctx, LU->numel());
  phi::funcs::TrilTriuCompute<T> tril_computer(LU->data<T>(), -1, true, H, W,
                                               L_dataptr);
  x_for_range(tril_computer);

  phi::funcs::TrilTriuCompute<T> triu_computer(
      LU->data<T>(), 0, false, H, W, U->mutable_data<T>(dev_ctx.GetPlace()));
  x_for_range(triu_computer);

  // set L's diagonal 1
  auto dim = std::min(H, W);
  framework::Tensor rowtensor, rt_dev;
  auto batchsize = product(phi::slice_ddim(udims, 0, udims.size() - 2));
  batchsize = std::max(static_cast<int>(batchsize), 1);
  arange<DeviceContext>(dev_ctx, &rowtensor, dim, batchsize, H);
  auto idtptr = rowtensor.data<int32_t>();
  if (platform::is_gpu_place(dev_ctx.GetPlace())) {
    framework::TensorCopy(rowtensor, dev_ctx.GetPlace(), &rt_dev);
    idtptr = rt_dev.data<int32_t>();
  }

  platform::ForRange<DeviceContext> for_range(dev_ctx, rowtensor.numel());
  OneFunctor<T> functor(L_dataptr, idtptr, W, dim);
  for_range(functor);
}

template <typename DeviceContext, typename T>
void scatterpivot(const DeviceContext& dev_ctx, T* out_data,
                  framework::Tensor* idlst, int w, int dim) {
  framework::Tensor idlst_tmp;
  idlst_tmp.Resize(idlst->dims());
  idlst_tmp.mutable_data<int32_t>(dev_ctx.GetPlace());
  framework::TensorCopy(*idlst, dev_ctx.GetPlace(), &idlst_tmp);
  auto idtptr = idlst_tmp.data<int32_t>();

  platform::ForRange<DeviceContext> for_range(dev_ctx, idlst_tmp.numel());
  OneFunctor<T> functor(out_data, idtptr, w, dim);
  for_range(functor);
}

template <typename DeviceContext, typename T>
void Unpack_Pivot(const DeviceContext& dev_ctx, const framework::Tensor& Pivot,
                  framework::Tensor* P, int h, int w) {
  auto dims = Pivot.dims();
  auto Pdimvec = vectorize(dims);
  auto prank = Pdimvec.size();
  auto Pnum = dims[prank - 1];
  framework::Tensor Pivot_cpu;
  platform::CPUPlace cpu;
  framework::TensorCopy(Pivot, cpu, &Pivot_cpu);
  auto pdataptr = Pivot_cpu.data<int32_t>();
  Pdimvec[prank - 1] = h;
  Pdimvec.emplace_back(h);
  auto Pdim = phi::make_ddim(Pdimvec);
  P->Resize(Pdim);
  auto pdata = P->mutable_data<T>(dev_ctx.GetPlace());
  phi::funcs::SetConstant<DeviceContext, T> setter;
  setter(dev_ctx, P, static_cast<T>(0));

  auto batchsize = product(phi::slice_ddim(dims, 0, prank - 1));
  batchsize = std::max(static_cast<int>(batchsize), 1);
  framework::Tensor idt;
  for (int i = 0; i < batchsize; i++) {
    arange<DeviceContext>(dev_ctx, &idt, h);
    auto idlst = idt.data<int32_t>();
    for (int j = 0; j < Pnum; j++) {
      if (idlst[pdataptr[i * Pnum + j] - 1] == idlst[j]) continue;
      auto temp = idlst[j];
      idlst[j] = idlst[pdataptr[i * Pnum + j] - 1];
      idlst[pdataptr[i * Pnum + j] - 1] = temp;
    }
    scatterpivot(dev_ctx, &(pdata[i * h * h]), &idt, h, h);
  }
}

template <typename DeviceContext, typename T>
class LUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto xin = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Input<framework::Tensor>("Out");
    auto P = ctx.Input<framework::Tensor>("Pivots");
    auto dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    const auto& dev_ctx = ctx.template device_context<DeviceContext>();
    math::DeviceIndependenceTensorOperations<DeviceContext, T> helper(ctx);
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);

    auto xdims = xin->dims();
    int xrank = xdims.size();
    int64_t m = xdims[xrank - 2];
    int64_t n = xdims[xrank - 1];
    int64_t k = std::min(m, n);

    framework::Tensor L, U, L_narrow, U_narrow, L_narrow_mH, U_narrow_mH,
        grad_narrow;
    LU_Unpack<DeviceContext, T>(dev_ctx, out, &L, &U);

    Tensor_narrow<DeviceContext, T>(ctx, &L, &L_narrow, 0, k, 0, k);
    Tensor_narrow<DeviceContext, T>(ctx, &U, &U_narrow, 0, k, 0, k);
    Tensor_narrow<DeviceContext, T>(ctx, dout, &grad_narrow, 0, k, 0, k);
    auto graddims = grad_narrow.dims();

    Tensor_Conj<DeviceContext, T>(dev_ctx, L_narrow, &L_narrow_mH);
    Tensor_Conj<DeviceContext, T>(dev_ctx, U_narrow, &U_narrow_mH);
    L_narrow_mH = helper.Transpose(L_narrow_mH);
    U_narrow_mH = helper.Transpose(U_narrow_mH);

    auto LmHdims = L_narrow_mH.dims();
    auto UmHdims = U_narrow_mH.dims();

    framework::Tensor phi_L, phi_U, phi, psi;
    phi_L.Resize(LmHdims);
    phi_L.mutable_data<T>(ctx.GetPlace());
    phi_U.Resize(UmHdims);
    phi_U.mutable_data<T>(ctx.GetPlace());
    auto mat_dim_l = phi::funcs::CreateMatrixDescriptor(LmHdims, 0, false);
    auto mat_dim_u = phi::funcs::CreateMatrixDescriptor(UmHdims, 0, false);
    auto mat_dim_g = phi::funcs::CreateMatrixDescriptor(graddims, 0, false);
    blas.MatMul(L_narrow_mH, mat_dim_l, grad_narrow, mat_dim_g,
                static_cast<T>(1), &phi_L, static_cast<T>(0));

    blas.MatMul(grad_narrow, mat_dim_g, U_narrow_mH, mat_dim_u,
                static_cast<T>(1), &phi_U, static_cast<T>(0));

    auto phil_rank = LmHdims.size();
    auto phiu_rank = UmHdims.size();
    platform::ForRange<DeviceContext> l_for_range(dev_ctx, phi_L.numel());
    phi::funcs::TrilTriuCompute<T> tril_computer(
        phi_L.data<T>(), -1, true, LmHdims[phil_rank - 2],
        LmHdims[phil_rank - 1], phi_L.data<T>());
    l_for_range(tril_computer);

    platform::ForRange<DeviceContext> u_for_range(dev_ctx, phi_U.numel());
    phi::funcs::TrilTriuCompute<T> triu_computer(
        phi_U.data<T>(), 0, false, UmHdims[phiu_rank - 2],
        UmHdims[phiu_rank - 1], phi_U.data<T>());
    u_for_range(triu_computer);

    Tensor_Add<DeviceContext, T>(dev_ctx, phi_L, phi_U, &phi);
    psi.Resize(xdims);
    psi.mutable_data<T>(ctx.GetPlace());
    phi::funcs::SetConstant<DeviceContext, T> setter;
    setter(dev_ctx, &psi, static_cast<T>(0));

    std::vector<int64_t> axes = {xrank - 2, xrank - 1};
    std::vector<int64_t> slice_starts(2, 0);
    std::vector<int64_t> slice_ends(2, 0);
    auto valuedims = vectorize(xdims);

    framework::Tensor Pmat;
    Unpack_Pivot<DeviceContext, T>(dev_ctx, *P, &Pmat, m, k);

    using Context =
        typename framework::ConvertToPhiContext<DeviceContext>::TYPE;
    auto& phi_dev_ctx = static_cast<const Context&>(dev_ctx);

    if (m <= n) {
      if (k < n) {
        framework::Tensor U_complement, U_grad_complement, phi_complement,
            phi_complement_l;
        Tensor_narrow<DeviceContext, T>(ctx, &U, &U_complement, 0, k, k, n);
        Tensor_narrow<DeviceContext, T>(ctx, dout, &U_grad_complement, 0, k, k,
                                        n);
        framework::Tensor U_complement_mH = helper.Transpose(U_complement);

        Tensor_Conj<DeviceContext, T>(dev_ctx, U_complement_mH,
                                      &U_complement_mH);

        auto mat_dim_g = phi::funcs::CreateMatrixDescriptor(
            U_grad_complement.dims(), 0, false);
        auto mat_dim_u = phi::funcs::CreateMatrixDescriptor(
            U_complement_mH.dims(), 0, false);
        auto phidims = UmHdims;
        phidims[UmHdims.size() - 2] = k;
        phidims[UmHdims.size() - 1] = k;
        phi_complement.Resize(phidims);
        phi_complement.mutable_data<T>(ctx.GetPlace());
        blas.MatMul(U_grad_complement, mat_dim_g, U_complement_mH, mat_dim_u,
                    static_cast<T>(1), &phi_complement, static_cast<T>(0));

        phi_complement_l.Resize(phidims);
        phi_complement_l.mutable_data<T>(ctx.GetPlace());
        const auto H = phidims[phidims.size() - 2];
        const auto W = phidims[phidims.size() - 1];
        platform::ForRange<DeviceContext> x_for_range(dev_ctx,
                                                      phi_complement.numel());
        phi::funcs::TrilTriuCompute<T> tril_computer(
            phi_complement.data<T>(), -1, true, H, W,
            phi_complement_l.data<T>());
        x_for_range(tril_computer);

        Tensor_Sub<DeviceContext, T>(dev_ctx, phi, phi_complement_l, &phi);

        slice_starts[0] = 0;
        slice_starts[1] = k;
        slice_ends[0] = k;
        slice_ends[1] = n;
        valuedims[xrank - 2] = k;
        valuedims[xrank - 1] = n - k;
        SetValueCompute_dispatch<DeviceContext, T>(
            ctx, &psi, &U_grad_complement, &psi, axes, &slice_starts,
            &slice_ends, valuedims, xrank);
      }

      framework::Tensor psi_principal, phi_mH, psi_tmp;
      Tensor_Conj<DeviceContext, T>(dev_ctx, phi, &phi_mH);
      phi_mH = helper.Transpose(phi_mH);

      phi::TriangularSolveKernel<T, Context>(
          phi_dev_ctx, U_narrow, phi_mH, true, false, false, &psi_principal);

      Tensor_Conj<DeviceContext, T>(dev_ctx, psi_principal, &psi_principal);
      psi_principal = helper.Transpose(psi_principal);
      slice_starts[0] = 0;
      slice_starts[1] = 0;
      slice_ends[0] = k;
      slice_ends[1] = k;
      valuedims[xrank - 2] = k;
      valuedims[xrank - 1] = k;

      SetValueCompute_dispatch<DeviceContext, T>(ctx, &psi, &psi_principal,
                                                 &psi, axes, &slice_starts,
                                                 &slice_ends, valuedims, xrank);

      phi::TriangularSolveKernel<T, Context>(phi_dev_ctx, L_narrow_mH, psi,
                                             true, false, true, &psi_tmp);

      auto mat_dim_p =
          phi::funcs::CreateMatrixDescriptor(Pmat.dims(), 0, false);
      auto mat_dim_b =
          phi::funcs::CreateMatrixDescriptor(psi_tmp.dims(), 0, false);
      blas.MatMul(Pmat, mat_dim_p, psi_tmp, mat_dim_b, static_cast<T>(1), dx,
                  static_cast<T>(0));
    } else {
      framework::Tensor L_complement, L_grad_complement, phi_complement,
          phi_complement_u;
      Tensor_narrow<DeviceContext, T>(ctx, &L, &L_complement, k, m, 0, k);
      Tensor_narrow<DeviceContext, T>(ctx, dout, &L_grad_complement, k, m, 0,
                                      k);
      framework::Tensor L_complement_mH = helper.Transpose(L_complement);
      Tensor_Conj<DeviceContext, T>(dev_ctx, L_complement_mH, &L_complement_mH);

      auto mat_dim_g = phi::funcs::CreateMatrixDescriptor(
          L_grad_complement.dims(), 0, false);
      auto mat_dim_u =
          phi::funcs::CreateMatrixDescriptor(L_complement_mH.dims(), 0, false);
      auto phidims = LmHdims;
      phidims[LmHdims.size() - 2] = k;
      phidims[LmHdims.size() - 1] = k;
      phi_complement.Resize(phidims);
      phi_complement.mutable_data<T>(ctx.GetPlace());
      blas.MatMul(L_complement_mH, mat_dim_u, L_grad_complement, mat_dim_g,
                  static_cast<T>(1), &phi_complement, static_cast<T>(0));

      phi_complement_u.Resize(phidims);
      phi_complement_u.mutable_data<T>(ctx.GetPlace());
      const auto H = phidims[phidims.size() - 2];
      const auto W = phidims[phidims.size() - 1];
      platform::ForRange<DeviceContext> x_for_range(dev_ctx,
                                                    phi_complement.numel());
      phi::funcs::TrilTriuCompute<T> triu_computer(
          phi_complement.data<T>(), 0, false, H, W, phi_complement_u.data<T>());
      x_for_range(triu_computer);

      Tensor_Sub<DeviceContext, T>(dev_ctx, phi, phi_complement_u, &phi);

      slice_starts[0] = k;
      slice_starts[1] = 0;
      slice_ends[0] = m;
      slice_ends[1] = k;
      valuedims[xrank - 2] = m - k;
      valuedims[xrank - 1] = k;
      SetValueCompute_dispatch<DeviceContext, T>(ctx, &psi, &L_grad_complement,
                                                 &psi, axes, &slice_starts,
                                                 &slice_ends, valuedims, xrank);
      framework::Tensor psi_principal, phi_mH, psi_tmp, U_narrow_mH;

      phi::TriangularSolveKernel<T, Context>(phi_dev_ctx, L_narrow_mH, phi,
                                             true, false, true, &psi_principal);

      slice_starts[0] = 0;
      slice_starts[1] = 0;
      slice_ends[0] = k;
      slice_ends[1] = k;
      valuedims[xrank - 2] = k;
      valuedims[xrank - 1] = k;

      SetValueCompute_dispatch<DeviceContext, T>(ctx, &psi, &psi_principal,
                                                 &psi, axes, &slice_starts,
                                                 &slice_ends, valuedims, xrank);

      psi_tmp.Resize(psi.dims());
      psi_tmp.mutable_data<T>(ctx.GetPlace());
      auto mat_dim_p =
          phi::funcs::CreateMatrixDescriptor(Pmat.dims(), 0, false);
      auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(psi.dims(), 0, false);
      blas.MatMul(Pmat, mat_dim_p, psi, mat_dim_b, static_cast<T>(1), &psi_tmp,
                  static_cast<T>(0));
      psi_tmp = helper.Transpose(psi_tmp);

      Tensor_Conj<DeviceContext, T>(dev_ctx, U_narrow, &U_narrow_mH);
      phi::TriangularSolveKernel<T, Context>(phi_dev_ctx, U_narrow_mH, psi_tmp,
                                             true, false, false, &psi);
      *dx = helper.Transpose(psi);
    }
  }
};

}  // namespace operators
}  // namespace paddle
