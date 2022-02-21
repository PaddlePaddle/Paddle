#pragma once

#include <tuple>

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"

namespace paddle {
namespace experimental {


PADDLE_API Tensor add(const Tensor& x, const Tensor& y);

PADDLE_API Tensor cast(const Tensor& x, DataType out_dtype);

PADDLE_API Tensor concat(const std::vector<Tensor>& x, const Scalar& axis);

PADDLE_API Tensor conj(const Tensor& x);

PADDLE_API Tensor divide(const Tensor& x, const Tensor& y);

PADDLE_API Tensor dot(const Tensor& x, const Tensor& y);

PADDLE_API Tensor empty(const ScalarArray& shape, DataType dtype=DataType::FLOAT32, Backend place=Backend::CPU, DataLayout layout=DataLayout::NCHW);

PADDLE_API Tensor empty_like(const Tensor& x, DataType dtype=DataType::UNDEFINED, Backend place=Backend::UNDEFINED, DataLayout layout=DataLayout::UNDEFINED);

PADDLE_API Tensor flatten(const Tensor& x, int start_axis, int stop_axis);

PADDLE_API Tensor full(const ScalarArray& shape, const Scalar& value, DataType dtype=DataType::FLOAT32, Backend place=Backend::CPU, DataLayout layout=DataLayout::NCHW);

PADDLE_API Tensor full_like(const Tensor& x, const Scalar& value, DataType dtype=DataType::UNDEFINED, Backend place=Backend::UNDEFINED, DataLayout layout=DataLayout::UNDEFINED);

PADDLE_API Tensor matmul(const Tensor& x, const Tensor& y, bool transpose_x=false, bool transpose_y=false);

PADDLE_API Tensor mean(const Tensor& x, const std::vector<int64_t>& axis={}, bool keep_dim=false);

PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y);

PADDLE_API Tensor ones_like(const Tensor& x, DataType dtype=DataType::UNDEFINED, Backend place=Backend::UNDEFINED, DataLayout layout=DataLayout::UNDEFINED);

PADDLE_API Tensor reshape(const Tensor& x, const ScalarArray& shape);

PADDLE_API Tensor scale(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale);

PADDLE_API Tensor sign(const Tensor& x);

PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y);

PADDLE_API Tensor sum(const Tensor& x, const std::vector<int64_t>& axis={}, DataType dtype=DataType::UNDEFINED, bool keep_dim=false);

PADDLE_API Tensor zeros_like(const Tensor& x, DataType dtype=DataType::UNDEFINED, Backend place=Backend::UNDEFINED, DataLayout layout=DataLayout::UNDEFINED);


}  // namespace experimental
}  // namespace paddle
