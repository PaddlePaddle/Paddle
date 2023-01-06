#pragma once

#include "paddle/phi/common/scalar.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace prim {

using Tensor = paddle::experimental::Tensor;
using Scalar = paddle::experimental::Scalar;

template <typename T>
Tensor multiply(const Tensor& x, const Tensor& y);

template <typename T>
Tensor pow(const Tensor& x, const Scalar& y);

template <typename T>
Tensor scale(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale);

}  // namespace prim
}  // namespace paddle
