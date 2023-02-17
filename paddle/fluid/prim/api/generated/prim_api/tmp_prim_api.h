#pragma once

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/place.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace prim {

using Tensor = paddle::experimental::Tensor;
using Scalar = paddle::experimental::Scalar;
using IntArray = paddle::experimental::IntArray;
using DataType = paddle::experimental::DataType;

template <typename T>
Tensor divide(const Tensor& x, const Tensor& y);

template <typename T>
Tensor expand(const Tensor& x, const IntArray& shape);

template <typename T>
Tensor full(const IntArray& shape, const Scalar& value, DataType dtype = DataType::FLOAT32, const Place& place = CPUPlace());

template <typename T>
Tensor multiply(const Tensor& x, const Tensor& y);

template <typename T>
Tensor pow(const Tensor& x, const Scalar& y);

template <typename T>
Tensor reshape(const Tensor& x, const IntArray& shape);

template <typename T>
Tensor scale(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale);

template <typename T>
Tensor sum(const Tensor& x, const IntArray& axis = {}, DataType dtype = DataType::UNDEFINED, bool keepdim = false);

template <typename T>
Tensor exp(const Tensor& x);

template <typename T>
Tensor unsqueeze(const Tensor& x, const IntArray& axis = {});

}  // namespace prim
}  // namespace paddle
