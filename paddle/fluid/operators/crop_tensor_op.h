/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/strided_memcpy.h"

namespace paddle {
namespace operators {  // Internal

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using framework::Tensor;

inline std::vector<int> get_new_data(
    const std::vector<const Tensor*>& list_new_tensor) {
  // get tensor from
  std::vector<int> vec_new_data;
  for (size_t i = 0; i < list_new_tensor.size(); ++i) {
    auto tensor = list_new_tensor[i];
    PADDLE_ENFORCE_EQ(
        tensor->dims(), framework::make_ddim({1}),
        platform::errors::InvalidArgument(
            "The tensor's shape in list of Op(crop_tensor) should be [1], "
            "but the value received is %d.",
            tensor->dims()));
    if (platform::is_gpu_place(tensor->place())) {
      framework::Tensor temp;
      paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);

      vec_new_data.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
    } else {
      vec_new_data.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
    }
  }

  return vec_new_data;
}

static framework::DDim ValidateShape(const std::vector<int> shape,
                                     const std::vector<int> offsets,
                                     const framework::DDim& in_dims) {
  auto in_dim_size = in_dims.size();
  auto shape_size = shape.size();
  PADDLE_ENFORCE_EQ(
      in_dim_size, shape_size,
      platform::errors::InvalidArgument(
          "The number of elements (%d) for shape of Op(crop_tensor) should be "
          "equal to the number of dimensions (%d) of the input tensor.",
          shape_size, in_dim_size));
  std::vector<int64_t> output_shape(shape.size(), 0);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 && in_dims[i] > 0) {
      PADDLE_ENFORCE_NE(shape[i], 0,
                        platform::errors::InvalidArgument(
                            "The value (%d) of the %uth element for shape of "
                            "Op(crop_tensor) should not be zero.",
                            shape[i], i));
      PADDLE_ENFORCE_EQ(shape[i], -1, platform::errors::InvalidArgument(
                                          "When the value (%d) of the %uth "
                                          "element for shape of Op(crop_tensor)"
                                          " is negative, only -1 is supported.",
                                          shape[i], i));
      output_shape[i] = in_dims[i] - offsets[i];
    } else {
      output_shape[i] = static_cast<int64_t>(shape[i]);
    }
  }

  return framework::make_ddim(output_shape);
}

static std::vector<int> GetShape(const framework::ExecutionContext& ctx) {
  std::vector<int> res;
  int rank = ctx.Input<Tensor>("X")->dims().size();
  auto list_new_shape_tensor = ctx.MultiInput<framework::Tensor>("ShapeTensor");
  if (list_new_shape_tensor.size() > 0) {
    // have offsets tensor list
    PADDLE_ENFORCE_EQ(
        list_new_shape_tensor.size(), rank,
        platform::errors::InvalidArgument(
            "The number of tensors (%d) for the input ShapeTensor of "
            "Op(crop_tensor) must be equal to the number of "
            "dimensions (%d) of the input.",
            list_new_shape_tensor.size(), rank));
    res = get_new_data(list_new_shape_tensor);

    return res;
  }

  auto* shape_tensor = ctx.HasInput("Shape")
                           ? ctx.Input<framework::LoDTensor>("Shape")
                           : nullptr;
  if (shape_tensor) {
    auto* shape_data = shape_tensor->data<int>();
    framework::Tensor cpu_shape_tensor;
    if (platform::is_gpu_place(shape_tensor->place())) {
      paddle::framework::TensorCopySync(*shape_tensor, platform::CPUPlace(),
                                        &cpu_shape_tensor);
      shape_data = cpu_shape_tensor.data<int>();
    }
    res = std::vector<int>(shape_data, shape_data + shape_tensor->numel());
  }

  return res;
}

static std::vector<int> GetOffsets(const framework::ExecutionContext& ctx) {
  std::vector<int> res;
  int rank = ctx.Input<Tensor>("X")->dims().size();
  auto list_new_offsets_tensor =
      ctx.MultiInput<framework::Tensor>("OffsetsTensor");
  if (list_new_offsets_tensor.size() > 0) {
    // have offsets tensor list
    res = get_new_data(list_new_offsets_tensor);

    return res;
  }

  if (ctx.HasInput("Offsets")) {
    const auto* offsets_tensor = ctx.Input<Tensor>("Offsets");
    PADDLE_ENFORCE_EQ(offsets_tensor->dims().size(), 1,
                      platform::errors::InvalidArgument(
                          "The number of dimensions of input 'Offsets' must "
                          "be 1, but the value received is: %d.",
                          offsets_tensor->dims().size()));
    PADDLE_ENFORCE_EQ(rank, offsets_tensor->dims()[0],
                      platform::errors::InvalidArgument(
                          "The number of elements (%d) for "
                          "input 'Offsets' must be equal to "
                          "the number of dimensions (%d) of the input tensor.",
                          offsets_tensor->dims()[0], rank));

    const int* offsets_data;
    framework::Tensor cpu_tmp_tensor;
    if (platform::is_cpu_place(offsets_tensor->place())) {
      offsets_data = offsets_tensor->data<int>();
    } else {
      framework::TensorCopySync(*offsets_tensor, platform::CPUPlace(),
                                &cpu_tmp_tensor);
      offsets_data = cpu_tmp_tensor.data<int>();
    }
    res = std::vector<int>(offsets_data, offsets_data + rank);
  } else {
    res = ctx.Attr<std::vector<int>>("offsets");
    PADDLE_ENFORCE_EQ(
        rank, static_cast<int>(res.size()),
        platform::errors::InvalidArgument("The number of elements (%d) for "
                                          "input 'Offsets' must be equal to "
                                          "the number of dimensions (%d) "
                                          "of the input tensor.",
                                          static_cast<int>(res.size()), rank));
  }
  return res;
}

template <typename DeviceContext, typename T, size_t D>
void CropTensorFunction(const framework::ExecutionContext& context) {
  auto* x = context.Input<Tensor>("X");
  auto* out = context.Output<Tensor>("Out");
  auto x_dims = x->dims();
  auto out_dims = out->dims();

  // get shape from Input(ShapeTensor) of Input(Shape)
  std::vector<int> shape = GetShape(context);
  // out_dims set by arrt(shape)
  if (shape.size() == 0) {
    for (int i = 0; i < out_dims.size(); ++i) {
      shape.push_back(out_dims[i]);
    }
  }

  auto offsets = GetOffsets(context);
  out_dims = ValidateShape(shape, offsets, x->dims());
  out->mutable_data<T>(out_dims, context.GetPlace());
  for (size_t i = 0; i < offsets.size(); ++i) {
    PADDLE_ENFORCE_LE(offsets[i] + shape[i], x_dims[i],
                      platform::errors::InvalidArgument(
                          "The sum of the %uth elements of "
                          "offsets (%d) and shape (%d) of Op(crop_tensor) "
                          "should be less than or "
                          "equal to the size of %uth dimension of the input.",
                          i, offsets[i], shape[i], i));
  }

  auto x_tensor = EigenTensor<T, D>::From(*x);
  auto out_tensor = EigenTensor<T, D>::From(*out);
  Eigen::DSizes<Eigen::DenseIndex, D> e_offsets;
  Eigen::DSizes<Eigen::DenseIndex, D> e_shape;
  for (size_t i = 0; i < D; ++i) {
    e_offsets[i] = offsets[i];
    e_shape[i] = out->dims()[i];
  }
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(
      place, out_tensor, x_tensor, e_offsets, e_shape);
}

template <typename DeviceContext, typename T>
class CropTensorKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    int rank = context.Input<Tensor>("X")->dims().size();
    PADDLE_ENFORCE_GE(
        rank, 1,
        platform::errors::InvalidArgument(
            "The number of dimensions of the input 'x' for "
            "Op(crop_tensor) must be greater than or equal to 1, but the "
            "value received is %d.",
            rank));
    PADDLE_ENFORCE_LE(
        rank, 6, platform::errors::InvalidArgument(
                     "The number of dimensions of the input 'x' for "
                     "Op(crop_tensor) must be less than or equal to 6, but the "
                     "value received is %d.",
                     rank));
    switch (rank) {
      case 1:
        CropTensorFunction<DeviceContext, T, 1>(context);
        break;
      case 2:
        CropTensorFunction<DeviceContext, T, 2>(context);
        break;
      case 3:
        CropTensorFunction<DeviceContext, T, 3>(context);
        break;
      case 4:
        CropTensorFunction<DeviceContext, T, 4>(context);
        break;
      case 5:
        CropTensorFunction<DeviceContext, T, 5>(context);
        break;
      case 6:
        CropTensorFunction<DeviceContext, T, 6>(context);
        break;
    }
  }
};

template <typename DeviceContext, typename T, size_t D>
void CropTensorGradFunction(const framework::ExecutionContext& context) {
  auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
  auto* x = context.Input<Tensor>("X");
  if (d_x != nullptr) {
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    d_x->mutable_data<T>(x->dims(), context.GetPlace());
    auto offsets = GetOffsets(context);
    Eigen::array<std::pair<int64_t, int64_t>, D> paddings;
    for (size_t i = 0; i < D; ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = d_x->dims()[i] - d_out->dims()[i] - offsets[i];
    }
    auto d_x_tensor = EigenTensor<T, D>::From(*d_x);
    auto d_out_tensor = EigenTensor<T, D>::From(*d_out);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
        place, d_x_tensor, d_out_tensor, paddings, static_cast<T>(0));
  }
}

template <typename DeviceContext, typename T>
class CropTensorGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    size_t rank =
        context.Input<Tensor>(framework::GradVarName("Out"))->dims().size();
    PADDLE_ENFORCE_GE(
        rank, 1,
        platform::errors::InvalidArgument(
            "The number of dimensions of the input 'Out@GRAD' for "
            "Op(crop_tensor_grad) must be greater than or equal to 1, but the "
            "value received is %d.",
            rank));
    PADDLE_ENFORCE_LE(
        rank, 6,
        platform::errors::InvalidArgument(
            "The number of dimensions of the input 'Out@GRAD' for "
            "Op(crop_tensor_grad) must be less than or equal to 6, but the "
            "value received is %d.",
            rank));
    switch (rank) {
      case 1:
        CropTensorGradFunction<DeviceContext, T, 1>(context);
        break;
      case 2:
        CropTensorGradFunction<DeviceContext, T, 2>(context);
        break;
      case 3:
        CropTensorGradFunction<DeviceContext, T, 3>(context);
        break;
      case 4:
        CropTensorGradFunction<DeviceContext, T, 4>(context);
        break;
      case 5:
        CropTensorGradFunction<DeviceContext, T, 5>(context);
        break;
      case 6:
        CropTensorGradFunction<DeviceContext, T, 6>(context);
        break;
    }
  }
};

}  // namespace operators
}  // namespace paddle
