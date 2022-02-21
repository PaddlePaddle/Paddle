#pragma once

#include <tuple>

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"

namespace paddle {
namespace experimental {


PADDLE_API std::vector<std::vector<Tensor>> matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x=false, bool transpose_y=false);

PADDLE_API Tensor scale_grad(const Tensor& out_grad, const Scalar& scale, float bias=0.0, bool bias_after_scale=true);


}  // namespace experimental
}  // namespace paddle
