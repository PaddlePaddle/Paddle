#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {
namespace strings {


// out@StringTensor

PADDLE_API Tensor empty(const IntArray& shape, const Place& place = CPUPlace());


// out@StringTensor

PADDLE_API Tensor empty_like(const Tensor& x, const Place& place = {});


// out@StringTensor

PADDLE_API Tensor lower(const Tensor& x, bool use_utf8_encoding);


// out@StringTensor

PADDLE_API Tensor upper(const Tensor& x, bool use_utf8_encoding);



}  // namespace strings
}  // namespace experimental
}  // namespace paddle
