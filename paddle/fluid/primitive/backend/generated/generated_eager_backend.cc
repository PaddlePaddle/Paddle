// Auto Generated, DO NOT EDIT!

#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/eager/api/manual/eager_manual/dygraph_forward_api.h"
#include "paddle/fluid/primitive/backend/generated/generated_backend.h"

namespace paddle {
namespace primitive {
namespace backend {

template <>
Tensor abs<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::abs_ad_func(x);
 
}

template <>
Tensor bitwise_and<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::bitwise_and_ad_func(x, y);
 
}

template <>
Tensor bitwise_not<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::bitwise_not_ad_func(x);
 
}

template <>
Tensor bitwise_or<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::bitwise_or_ad_func(x, y);
 
}

template <>
Tensor bitwise_xor<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::bitwise_xor_ad_func(x, y);
 
}

template <>
Tensor concat<Tensor>(const std::vector<Tensor>& x, const Scalar& axis) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::concat_ad_func(x, axis);
 
}

template <>
Tensor erf<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::erf_ad_func(x);
 
}

template <>
Tensor exp<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::exp_ad_func(x);
 
}

template <>
Tensor expand<Tensor>(const Tensor& x, const IntArray& shape) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::expand_ad_func(x, shape);
 
}

template <>
Tensor floor<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::floor_ad_func(x);
 
}

template <>
Tensor gather_nd<Tensor>(const Tensor& x, const Tensor& index) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::gather_nd_ad_func(x, index);
 
}

template <>
Tensor log<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::log_ad_func(x);
 
}

template <>
Tensor roll<Tensor>(const Tensor& x, const IntArray& shifts, const std::vector<int64_t>& axis) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::roll_ad_func(x, shifts, axis);
 
}

template <>
Tensor scale<Tensor>(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::scale_ad_func(x, scale, bias, bias_after_scale);
 
}

template <>
Tensor scatter<Tensor>(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::scatter_ad_func(x, index, updates, overwrite);
 
}

template <>
Tensor scatter_nd_add<Tensor>(const Tensor& x, const Tensor& index, const Tensor& updates) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::scatter_nd_add_ad_func(x, index, updates);
 
}

template <>
Tensor sqrt<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::sqrt_ad_func(x);
 
}

template <>
Tensor tanh<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::tanh_ad_func(x);
 
}

template <>
Tensor add<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::add_ad_func(x, y);
 
}

template <>
Tensor assign<Tensor>(const Tensor& x) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::assign_ad_func(x);
 
}

template <>
Tensor cast<Tensor>(const Tensor& x, DataType dtype) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::cast_ad_func(x, dtype);
 
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
Tensor equal<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::equal_ad_func(x, y);
 
}

template <>
Tensor full<Tensor>(const IntArray& shape, const Scalar& value, DataType dtype, Place place) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::full_ad_func(shape, value, dtype, place);
 
}

template <>
Tensor greater_equal<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::greater_equal_ad_func(x, y);
 
}

template <>
Tensor greater_than<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::greater_than_ad_func(x, y);
 
}

template <>
Tensor less_equal<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::less_equal_ad_func(x, y);
 
}

template <>
Tensor less_than<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::less_than_ad_func(x, y);
 
}

template <>
Tensor matmul<Tensor>(const Tensor& x, const Tensor& y, bool transpose_x, bool transpose_y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::matmul_ad_func(x, y, transpose_x, transpose_y);
 
}

template <>
Tensor max<Tensor>(const Tensor& x, const IntArray& axis, bool keepdim) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::max_ad_func(x, axis, keepdim);
 
}

template <>
Tensor maximum<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::maximum_ad_func(x, y);
 
}

template <>
Tensor minimum<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::minimum_ad_func(x, y);
 
}

template <>
Tensor multiply<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::multiply_ad_func(x, y);
 
}

template <>
Tensor not_equal<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::not_equal_ad_func(x, y);
 
}

template <>
Tensor reshape<Tensor>(const Tensor& x, const IntArray& shape) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::reshape_ad_func(x, shape);
 
}

template <>
std::vector<Tensor> split<Tensor>(const Tensor& x, const IntArray& sections, const Scalar& axis) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::split_ad_func(x, sections, axis);
 
}

template <>
Tensor subtract<Tensor>(const Tensor& x, const Tensor& y) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::subtract_ad_func(x, y);
 
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

template <>
Tensor transpose<Tensor>(const Tensor& x, const std::vector<int>& perm) {
  VLOG(4) << "Eager Prim API {name}_ad_func call";
  return ::transpose_ad_func(x, perm);
 
}


}  // namespace backend
}  // namespace primitive
}  // namespace paddle
