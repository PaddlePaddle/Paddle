// Auto Generated, DO NOT EDIT!

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/primitive/backend/generated/generated_backend.h"

namespace paddle {
namespace primitive {
namespace backend {

template <>
Tensor concat<Tensor>(const std::vector<Tensor>& x, const Scalar& axis) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::concat_ad_func(x, axis);
 
}

template <>
Tensor expand<Tensor>(const Tensor& x, const IntArray& shape) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::expand_ad_func(x, shape);
 
}

template <>
Tensor scale<Tensor>(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::scale_ad_func(x, scale, bias, bias_after_scale);
 
}

template <>
Tensor add<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::add_ad_func(x, y);
 
}

template <>
Tensor divide<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::divide_ad_func(x, y);
 
}

template <>
Tensor elementwise_pow<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::elementwise_pow_ad_func(x, y);
 
}

template <>
Tensor full<Tensor>(const IntArray& shape, const Scalar& value, DataType dtype, Place place) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::full_ad_func(shape, value, dtype, place);
 
}

template <>
Tensor multiply<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::multiply_ad_func(x, y);
 
}

template <>
Tensor sum<Tensor>(const Tensor& x, const IntArray& axis, DataType dtype, bool keepdim) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::sum_ad_func(x, axis, dtype, keepdim);
 
}

template <>
Tensor tile<Tensor>(const Tensor& x, const IntArray& repeat_times) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::tile_ad_func(x, repeat_times);
 
}


}  // namespace backend
}  // namespace primitive
}  // namespace paddle
