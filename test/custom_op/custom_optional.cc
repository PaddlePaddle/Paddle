// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WIdata_tHOUdata_t WARRANdata_tIES OR CONDIdata_tIONS OF ANY KIND, either
// express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <vector>

#include "paddle/extension.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

template <typename data_t>
void add_one_pointer(const data_t* x_data, data_t* out_data, int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    out_data[i] += x_data[i];
  }
}

template <typename data_t>
void add_two_pointers(const data_t* x_data,
                      const data_t* y_data,
                      data_t* out_data,
                      int64_t numel) {
  for (size_t i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] + y_data[i];
  }
}

/*
if (y) {
  out = x + y;
} else {
  out = x + x;
}
*/
std::vector<paddle::Tensor> AddForward(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& y) {  // NOLINT
  CHECK_INPUT(x);
  paddle::Tensor out = paddle::empty(x.shape(), x.dtype(), x.place());

  if (y) {
    out = x + y.get();
  } else {
    out = x + x;
  }
  return {out};
}

std::vector<paddle::DataType> AddInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::optional<paddle::DataType>& y_dtype) {
  if (y_dtype) {
    return {*y_dtype};
  }
  return {x_dtype};
}

std::vector<std::vector<int64_t>> AddInferShape(
    const std::vector<int64_t>& x_shape,
    const paddle::optional<std::vector<int64_t>>& y_shape) {
  if (y_shape) {
    return {*y_shape};
  }
  return {x_shape};
}

/*
if (y) {
  x_grad = out_grad;
} else {
  x_grad = out_grad + out_grad;
}
*/
std::vector<paddle::Tensor> AddBackward(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& y,
    const paddle::Tensor& out_grad) {  // NOLINT
  CHECK_INPUT(x);
  paddle::Tensor x_grad = paddle::zeros(x.shape(), x.dtype(), x.place());

  if (y) {
    x_grad = out_grad;
  } else {
    x_grad = out_grad + out_grad;
  }
  return {x_grad};
}

PD_BUILD_OP(custom_add)
    .Inputs({"X", paddle::Optional("Y")})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AddForward))
    .SetInferShapeFn(PD_INFER_SHAPE(AddInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AddInferDtype));

PD_BUILD_GRAD_OP(custom_add)
    .Inputs({"X", paddle::Optional("Y"), paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(AddBackward));

/*
if (y) {
  out = x + y[0] + y[1] + ...;
} else {
  out = x + x;
}
*/
std::vector<paddle::Tensor> AddVectorForward(
    const paddle::Tensor& x,
    const paddle::optional<std::vector<paddle::Tensor>>& y) {  // NOLINT
  CHECK_INPUT(x);
  paddle::Tensor out = paddle::zeros(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "AddVectorForward", ([&] {
        if (y) {
          add_one_pointer<data_t>(
              x.data<data_t>(), out.data<data_t>(), out.size());
          for (size_t i = 0; i < y->size(); ++i) {
            add_one_pointer<data_t>(
                y->at(i).data<data_t>(), out.data<data_t>(), out.size());
          }
        } else {
          add_two_pointers<data_t>(
              x.data<data_t>(), x.data<data_t>(), out.data<data_t>(), x.size());
        }
      }));
  return {out};
}

std::vector<paddle::DataType> AddVectorInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::optional<std::vector<paddle::DataType>>& y_dtype) {
  if (y_dtype) {
    return {y_dtype->at(0)};
  }
  return {x_dtype};
}

std::vector<std::vector<int64_t>> AddVectorInferShape(
    const std::vector<int64_t>& x_shape,
    const paddle::optional<std::vector<std::vector<int64_t>>>& y_shape) {
  if (y_shape) {
    return {y_shape->at(0)};
  }
  return {x_shape};
}

/*
if (y) {
  x_grad = out_grad;
} else {
  x_grad = out_grad + out_grad;
}
*/
std::vector<paddle::Tensor> AddVectorBackward(
    const paddle::Tensor& x,
    const paddle::optional<std::vector<paddle::Tensor>>& y,
    const paddle::Tensor& out_grad) {  // NOLINT
  CHECK_INPUT(x);

  paddle::Tensor x_grad = paddle::zeros(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      out_grad.type(), "AddVectorBackward", ([&] {
        add_one_pointer<data_t>(
            out_grad.data<data_t>(), x_grad.data<data_t>(), out_grad.size());
        if (!y) {
          add_one_pointer<data_t>(
              out_grad.data<data_t>(), x_grad.data<data_t>(), out_grad.size());
        }
      }));

  return {x_grad};
}

PD_BUILD_OP(custom_add_vec)
    .Inputs({"X", paddle::Optional(paddle::Vec("Y"))})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AddVectorForward))
    .SetInferShapeFn(PD_INFER_SHAPE(AddVectorInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AddVectorInferDtype));

PD_BUILD_GRAD_OP(custom_add_vec)
    .Inputs({"X", paddle::Optional(paddle::Vec("Y")), paddle::Grad("Out")})
    .Outputs({paddle::Grad("X")})
    .SetKernelFn(PD_KERNEL(AddVectorBackward));

/*
if (y) {
  outX = 2 * x + y;
  outY = x + y;
} else {
  outX = 2 * x;
  outY = None;
}
*/
std::vector<paddle::Tensor> AddOptionalInplaceForward(
    const paddle::Tensor& x,
    paddle::optional<paddle::Tensor>& y) {  // NOLINT
  CHECK_INPUT(x);
  paddle::Tensor outX = paddle::zeros(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "AddOptionalInplaceForward", ([&] {
        add_two_pointers<data_t>(
            x.data<data_t>(), x.data<data_t>(), outX.data<data_t>(), x.size());
        if (y) {
          add_one_pointer<data_t>(
              y->data<data_t>(), outX.data<data_t>(), outX.size());
          add_one_pointer<data_t>(
              x.data<data_t>(), y->data<data_t>(), x.size());
        }
      }));
  // No need to return y, because we set it as inplace input.
  return {outX};
}

std::vector<paddle::DataType> AddOptionalInplaceInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::optional<paddle::DataType>& y_dtype) {
  return {x_dtype};
}

std::vector<std::vector<int64_t>> AddOptionalInplaceInferShape(
    const std::vector<int64_t>& x_shape,
    const paddle::optional<std::vector<int64_t>>& y_shape) {
  return {x_shape};
}

/*
if (y) {
  x_grad = outX_grad * 2 + outY_grad;
  y_grad = outX_grad + outY_grad;
} else {
  x_grad = outX_grad * 2;
  y_grad = None;
}
*/
std::vector<paddle::Tensor> AddOptionalInplaceBackward(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& y,
    const paddle::Tensor& outx_grad,
    paddle::optional<paddle::Tensor>& outy_grad) {  // NOLINT
  CHECK_INPUT(x);

  paddle::Tensor x_grad = paddle::zeros(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      outx_grad.type(), "AddOptionalInplaceBackward", ([&] {
        add_two_pointers<data_t>(outx_grad.data<data_t>(),
                                 outx_grad.data<data_t>(),
                                 x_grad.data<data_t>(),
                                 x_grad.size());
        if (outy_grad) {
          add_one_pointer<data_t>(
              outy_grad->data<data_t>(), x_grad.data<data_t>(), x_grad.size());
          add_one_pointer<data_t>(outx_grad.data<data_t>(),
                                  outy_grad->data<data_t>(),
                                  outx_grad.size());
        }
      }));

  return {x_grad};
}

std::vector<std::vector<int64_t>> AddOptionalInplaceBackwardInferShape(
    const std::vector<int64_t>& x_shape,
    const paddle::optional<std::vector<int64_t>>& y_shape,
    const std::vector<int64_t>& x_grad_shape,
    const paddle::optional<std::vector<int64_t>>& y_grad_shape) {
  return {x_shape};
}

PD_BUILD_OP(custom_optional_inplace_add)
    .Inputs({"X", paddle::Optional("Y")})
    .Outputs({"OutX", paddle::Optional("OutY")})
    .SetInplaceMap({{paddle::Optional("Y"), paddle::Optional("OutY")}})
    .SetKernelFn(PD_KERNEL(AddOptionalInplaceForward))
    .SetInferShapeFn(PD_INFER_SHAPE(AddOptionalInplaceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AddOptionalInplaceInferDtype));

PD_BUILD_GRAD_OP(custom_optional_inplace_add)
    .Inputs({"X",
             paddle::Optional("Y"),
             paddle::Grad("OutX"),
             paddle::Grad(paddle::Optional("OutY"))})
    .Outputs({paddle::Grad("X"), paddle::Grad(paddle::Optional("Y"))})
    .SetInplaceMap({{paddle::Grad(paddle::Optional("OutY")),
                     paddle::Grad(paddle::Optional("Y"))}})
    .SetKernelFn(PD_KERNEL(AddOptionalInplaceBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(AddOptionalInplaceBackwardInferShape));

/*
if (y) {
  outX = 2 * x + y[1...n];
  outY[i] = x + y[i];
} else {
  outX = 2 * x;
  outY = None;
}
*/
std::vector<paddle::Tensor> AddOptionalInplaceVectorForward(
    const paddle::Tensor& x,
    paddle::optional<std::vector<paddle::Tensor>>& y) {  // NOLINT
  CHECK_INPUT(x);
  paddle::Tensor outX = paddle::zeros(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      x.type(), "AddOptionalInplaceVectorForward", ([&] {
        add_two_pointers<data_t>(
            x.data<data_t>(), x.data<data_t>(), outX.data<data_t>(), x.size());
        if (y) {
          for (size_t i = 0; i < y->size(); ++i) {
            add_one_pointer<data_t>(
                y->at(i).data<data_t>(), outX.data<data_t>(), outX.size());
            add_one_pointer<data_t>(
                x.data<data_t>(), y->at(i).data<data_t>(), x.size());
          }
        }
      }));
  // No need to return y, because we set it as inplace input.
  return {outX};
}

std::vector<paddle::DataType> AddOptionalInplaceVectorInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::optional<std::vector<paddle::DataType>>& y_dtype) {
  return {x_dtype};
}

std::vector<std::vector<int64_t>> AddOptionalInplaceVectorInferShape(
    const std::vector<int64_t>& x_shape,
    const paddle::optional<std::vector<std::vector<int64_t>>>& y_shape) {
  return {x_shape};
}

/*
if (outy_grad) {
  x_grad = outX_grad * 2 + outY_grad[1...n];
  y_grad[i] = outX_grad + outY_grad[i];
} else {
  x_grad = outX_grad * 2;
  y_grad = None;
}
*/
std::vector<paddle::Tensor> AddOptionalInplaceVectorBackward(
    const paddle::Tensor& x,
    const paddle::optional<std::vector<paddle::Tensor>>& y,
    const paddle::Tensor& outx_grad,
    paddle::optional<std::vector<paddle::Tensor>>& outy_grad) {  // NOLINT
  CHECK_INPUT(x);

  paddle::Tensor x_grad = paddle::zeros(x.shape(), x.dtype(), x.place());

  PD_DISPATCH_FLOATING_TYPES(
      outx_grad.type(), "AddOptionalInplaceVectorBackward", ([&] {
        add_two_pointers<data_t>(outx_grad.data<data_t>(),
                                 outx_grad.data<data_t>(),
                                 x_grad.data<data_t>(),
                                 x_grad.size());
        if (outy_grad) {
          for (size_t i = 0; i < outy_grad->size(); ++i) {
            add_one_pointer<data_t>(outy_grad->at(i).data<data_t>(),
                                    x_grad.data<data_t>(),
                                    x_grad.size());
            add_one_pointer<data_t>(outx_grad.data<data_t>(),
                                    outy_grad->at(i).data<data_t>(),
                                    outx_grad.size());
          }
        }
      }));

  return {x_grad};
}

std::vector<std::vector<int64_t>> AddOptionalInplaceVectorBackwardInferShape(
    const std::vector<int64_t>& x_shape,
    const paddle::optional<std::vector<std::vector<int64_t>>>& y_shape,
    const std::vector<int64_t>& x_grad_shape,
    const paddle::optional<std::vector<std::vector<int64_t>>>& y_grad_shape) {
  return {x_shape};
}

PD_BUILD_OP(custom_optional_inplace_add_vec)
    .Inputs({"X", paddle::Optional(paddle::Vec("Y"))})
    .Outputs({"OutX", paddle::Optional(paddle::Vec("OutY"))})
    .SetInplaceMap({{paddle::Optional(paddle::Vec("Y")),
                     paddle::Optional(paddle::Vec("OutY"))}})
    .SetKernelFn(PD_KERNEL(AddOptionalInplaceVectorForward))
    .SetInferShapeFn(PD_INFER_SHAPE(AddOptionalInplaceVectorInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AddOptionalInplaceVectorInferDtype));

PD_BUILD_GRAD_OP(custom_optional_inplace_add_vec)
    .Inputs({"X",
             paddle::Optional(paddle::Vec("Y")),
             paddle::Grad("OutX"),
             paddle::Grad(paddle::Optional(paddle::Vec("OutY")))})
    .Outputs({paddle::Grad("X"),
              paddle::Grad(paddle::Optional(paddle::Vec("Y")))})
    .SetInplaceMap({{paddle::Grad(paddle::Optional(paddle::Vec("OutY"))),
                     paddle::Grad(paddle::Optional(paddle::Vec("Y")))}})
    .SetKernelFn(PD_KERNEL(AddOptionalInplaceVectorBackward))
    .SetInferShapeFn(
        PD_INFER_SHAPE(AddOptionalInplaceVectorBackwardInferShape));
