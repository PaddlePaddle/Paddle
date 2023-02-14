
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/prim/api/generated/prim_api/prim_api.h"

namespace paddle {
namespace prim {

template <>
Tensor divide<Tensor>(const Tensor& x, const Tensor& y)
{
VLOG(4) << "Eager Prim API divide_ad_func call";
return ::divide_ad_func(x, y);
}

template <>
Tensor expand<Tensor>(const Tensor& x, const IntArray& shape)
{
VLOG(4) << "Eager Prim API expand_ad_func call";
return ::expand_ad_func(x, shape);
}

template <>
Tensor full<Tensor>(const IntArray& shape, const Scalar& value, DataType dtype, const Place& place)
{
VLOG(4) << "Eager Prim API full_ad_func call";
return ::full_ad_func(shape, value, dtype, place);
}

template <>
Tensor multiply<Tensor>(const Tensor& x, const Tensor& y)
{
VLOG(4) << "Eager Prim API multiply_ad_func call";
return ::multiply_ad_func(x, y);
}

template <>
Tensor pow<Tensor>(const Tensor& x, const Scalar& y)
{
VLOG(4) << "Eager Prim API pow_ad_func call";
return ::pow_ad_func(x, y);
}

template <>
Tensor reshape<Tensor>(const Tensor& x, const IntArray& shape)
{
VLOG(4) << "Eager Prim API reshape_ad_func call";
return ::reshape_ad_func(x, shape);
}

template <>
Tensor scale<Tensor>(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale)
{
VLOG(4) << "Eager Prim API scale_ad_func call";
return ::scale_ad_func(x, scale, bias, bias_after_scale);
}

template <>
Tensor sum<Tensor>(const Tensor& x, const IntArray& axis, DataType dtype, bool keepdim)
{
VLOG(4) << "Eager Prim API sum_ad_func call";
return ::sum_ad_func(x, axis, dtype, keepdim);
}

template <>
Tensor exp<Tensor>(const Tensor& x)
{
VLOG(4) << "Eager Prim API exp_ad_func call";
return ::exp_ad_func(x);
}

template <>
Tensor unsqueeze<Tensor>(const Tensor& x, const IntArray& axis)
{
VLOG(4) << "Eager Prim API unsqueeze_ad_func call";
return ::unsqueeze_ad_func(x, axis);
}

}  // namespace prim
}  // namespace paddle
