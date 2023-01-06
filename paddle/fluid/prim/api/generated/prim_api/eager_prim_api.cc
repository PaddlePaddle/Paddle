
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/api/generated/eager_generated/forwards/dygraph_functions.h"
#include "paddle/fluid/prim/api/generated/prim_api/prim_api.h"


namespace paddle {
namespace prim {

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
Tensor scale<Tensor>(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale)
{
VLOG(4) << "Eager Prim API scale_ad_func call";
return ::scale_ad_func(x, scale, bias, bias_after_scale);
 }

}  // namespace prim
}  // namespace paddle
