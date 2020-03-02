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
        "The tensor's shape in list of Op(crop_tensor) should be [1].");
    if (platform::is_gpu_place(tensor->place())) {
      framework::Tensor temp;
      TensorCopySync(*tensor, platform::CPUPlace(), &temp);

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
      "Attr(shape)'s size of Op(crop_tensor) should be equal "
      "to that of input Tensor. "
      "Please check the Attr(shape)'s size of Op(fluid.layers.crop_tensor).");
  std::vector<int64_t> output_shape(shape.size(), 0);
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0 && in_dims[i] > 0) {
      PADDLE_ENFORCE_NE(
          shape[i], 0,
          "The element in Attr(shape) of Op(crop_tensor) should not be zero.");
      PADDLE_ENFORCE_EQ(shape[i], -1,
                        "When the element in Attr(shape) of Op(crop_tensor) is "
                        "negative, only -1 is supported.");
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
    PADDLE_ENFORCE_EQ(list_new_shape_tensor.size(), rank,
                      "Input(ShapeTensor)'s length of Op(crop_tensor) should "
                      "be equal to dimension size of input tensor.");
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
      TensorCopySync(*shape_tensor, platform::CPUPlace(), &cpu_shape_tensor);
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
    PADDLE_ENFORCE_EQ(
        ctx.Attr<std::vector<int>>("offsets").empty(), true,
        "Input 'Offsets' and attribute 'offsets' should not be used "
        "at the same time.");
    const auto* offsets_tensor = ctx.Input<Tensor>("Offsets");
    PADDLE_ENFORCE_EQ(offsets_tensor->dims().size(), 1);
    PADDLE_ENFORCE_EQ(
        rank, offsets_tensor->dims()[0],
        "Offsets size should be equal to dimension size of input tensor.");
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
        "Offsets size should be equal to dimension size of input tensor.");
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
    PADDLE_ENFORCE_LE(
        offsets[i] + shape[i], x_dims[i],
        "The sum of the Attr(offsets) and Attr(shape) of Op(crop_tensor) "
        "should be less than or equal to corresponding input dimension size.");
  }

  auto x_tensor = EigenTensor<T, D>::From(*x);
  auto out_tensor = EigenTensor<T, D>::From(*out);
  Eigen::array<int, D> e_offsets;
  Eigen::array<int, D> e_shape;
  for (size_t i = 0; i < D; ++i) {
    e_offsets[i] = offsets[i];
    e_shape[i] = out->dims()[i];
  }
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  out_tensor.device(place) = x_tensor.slice(e_offsets, e_shape);
}

template <typename DeviceContext, typename T>
class CropTensorKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    int rank = context.Input<Tensor>("X")->dims().size();
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
      default:
        PADDLE_THROW(
            "CropTensorOp only support tensors with no more than 6 "
            "dimensions.");
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
    Eigen::array<std::pair<int, int>, D> paddings;
    for (size_t i = 0; i < D; ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = d_x->dims()[i] - d_out->dims()[i] - offsets[i];
    }
    auto d_x_tensor = EigenTensor<T, D>::From(*d_x);
    auto d_out_tensor = EigenTensor<T, D>::From(*d_out);
    d_x_tensor.device(
        *context.template device_context<DeviceContext>().eigen_device()) =
        d_out_tensor.pad(paddings, 0);
  }
}

template <typename DeviceContext, typename T>
class CropTensorGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    size_t rank =
        context.Input<Tensor>(framework::GradVarName("Out"))->dims().size();
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
      default:
        PADDLE_THROW(
            "CropTensorOp only support tensors with no more than 6 "
            "dimensions.");
    }
  }
};

}  // namespace operators
}  // namespace paddle
