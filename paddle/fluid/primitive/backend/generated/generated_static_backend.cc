// Auto Generated, DO NOT EDIT!

#include "paddle/fluid/primitive/backend/generated/generated_backend.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"

namespace paddle {
namespace primitive {
namespace backend {

using LazyTensor = paddle::primitive::LazyTensor;

template <>
Tensor abs<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::abs(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bitwise_and<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::bitwise_and(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bitwise_not<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::bitwise_not(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bitwise_or<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::bitwise_or(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bitwise_xor<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::bitwise_xor(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor concat<LazyTensor>(const std::vector<Tensor>& x, const Scalar& axis) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::concat(x_res, axis.to<int>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor concat<LazyTensor>(const std::vector<Tensor>& x, const Tensor& axis_) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::concat(x_res, axis_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor erf<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::erf(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor exp<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::exp(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor expand<LazyTensor>(const Tensor& x, const IntArray& shape) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::expand(x_res, shape.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor expand<LazyTensor>(const Tensor& x, const Tensor& shape_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult shape_res = std::static_pointer_cast<LazyTensor>(shape_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::expand(x_res, shape_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor floor<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::floor(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor gather_nd<LazyTensor>(const Tensor& x, const Tensor& index) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  auto op_res = paddle::dialect::gather_nd(x_res, index_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor log<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::log(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor roll<LazyTensor>(const Tensor& x, const IntArray& shifts, const std::vector<int64_t>& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::roll(x_res, shifts.GetData(), axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor roll<LazyTensor>(const Tensor& x, const Tensor& shifts_, const std::vector<int64_t>& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult shifts_res = std::static_pointer_cast<LazyTensor>(shifts_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::roll(x_res, shifts_res, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor scale<LazyTensor>(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::scale(x_res, scale.to<float>(), bias, bias_after_scale);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor scale<LazyTensor>(const Tensor& x, const Tensor& scale_, float bias, bool bias_after_scale) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult scale_res = std::static_pointer_cast<LazyTensor>(scale_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::scale(x_res, scale_res, bias, bias_after_scale);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor scatter<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value updates_res = std::static_pointer_cast<LazyTensor>(updates.impl())->value();
  auto op_res = paddle::dialect::scatter(x_res, index_res, updates_res, overwrite);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor scatter_nd_add<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& updates) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value updates_res = std::static_pointer_cast<LazyTensor>(updates.impl())->value();
  auto op_res = paddle::dialect::scatter_nd_add(x_res, index_res, updates_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sqrt<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sqrt(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor tanh<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::tanh(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor add<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::add(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor add_n<LazyTensor>(const std::vector<Tensor>& inputs) {
  std::vector<pir::Value> inputs_res(inputs.size());
  std::transform(inputs.begin(), inputs.end(), inputs_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::add_n(inputs_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor assign<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::assign(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cast<LazyTensor>(const Tensor& x, DataType dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cast(x_res, dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor divide<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::divide(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor elementwise_pow<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::elementwise_pow(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor equal<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::equal(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor full<LazyTensor>(const IntArray& shape, const Scalar& value, DataType dtype, Place place) {
  auto op_res = paddle::dialect::full(shape.GetData(), value.to<float>(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor greater_equal<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::greater_equal(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor greater_than<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::greater_than(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor less_equal<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::less_equal(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor less_than<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::less_than(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor matmul<LazyTensor>(const Tensor& x, const Tensor& y, bool transpose_x, bool transpose_y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::matmul(x_res, y_res, transpose_x, transpose_y);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor max<LazyTensor>(const Tensor& x, const IntArray& axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::max(x_res, axis.GetData(), keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor max<LazyTensor>(const Tensor& x, const Tensor& axis_, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::max(x_res, axis_res, keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor maximum<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::maximum(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor mean<LazyTensor>(const Tensor& x, const IntArray& axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::mean(x_res, axis.GetData(), keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor minimum<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::minimum(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor multiply<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::multiply(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor not_equal<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::not_equal(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor reshape<LazyTensor>(const Tensor& x, const IntArray& shape) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::reshape(x_res, shape.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor reshape<LazyTensor>(const Tensor& x, const Tensor& shape_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult shape_res = std::static_pointer_cast<LazyTensor>(shape_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::reshape(x_res, shape_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor slice<LazyTensor>(const Tensor& input, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::slice(input_res, axes, starts.GetData(), ends.GetData(), infer_flags, decrease_axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor slice<LazyTensor>(const Tensor& input, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::OpResult starts_res = std::static_pointer_cast<LazyTensor>(starts_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult ends_res = std::static_pointer_cast<LazyTensor>(ends_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::slice(input_res, starts_res, ends_res, axes, infer_flags, decrease_axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::vector<Tensor> split<LazyTensor>(const Tensor& x, const IntArray& sections, const Scalar& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::split(x_res, sections.GetData(), axis.to<int>());
  std::vector<Tensor> out(op_res.size());
  std::transform(op_res.begin(), op_res.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return out; 
}

template <>
std::vector<Tensor> split<LazyTensor>(const Tensor& x, const Tensor& sections_, const Tensor& axis_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult sections_res = std::static_pointer_cast<LazyTensor>(sections_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::split(x_res, sections_res, axis_res);
  std::vector<Tensor> out(op_res.size());
  std::transform(op_res.begin(), op_res.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return out; 
}

template <>
Tensor subtract<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::subtract(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sum<LazyTensor>(const Tensor& x, const IntArray& axis, DataType dtype, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sum(x_res, axis.GetData(), dtype, keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sum<LazyTensor>(const Tensor& x, const Tensor& axis_, DataType dtype, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::sum(x_res, axis_res, dtype, keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor tile<LazyTensor>(const Tensor& x, const IntArray& repeat_times) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::tile(x_res, repeat_times.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor tile<LazyTensor>(const Tensor& x, const Tensor& repeat_times_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult repeat_times_res = std::static_pointer_cast<LazyTensor>(repeat_times_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::tile(x_res, repeat_times_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor transpose<LazyTensor>(const Tensor& x, const std::vector<int>& perm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::transpose(x_res, perm);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor uniform<LazyTensor>(const IntArray& shape, DataType dtype, const Scalar& min, const Scalar& max, int seed, Place place) {
  auto op_res = paddle::dialect::uniform(shape.GetData(), dtype, min.to<float>(), max.to<float>(), seed, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor uniform<LazyTensor>(const Tensor& shape_, const Tensor& min_, const Tensor& max_, DataType dtype, int seed, Place place) {
  pir::OpResult shape_res = std::static_pointer_cast<LazyTensor>(shape_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult min_res = std::static_pointer_cast<LazyTensor>(min_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult max_res = std::static_pointer_cast<LazyTensor>(max_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::uniform(shape_res, min_res, max_res, dtype, seed, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::vector<Tensor> concat_grad<LazyTensor>(const std::vector<Tensor>& x, const Tensor& out_grad, const Scalar& axis) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::concat_grad(x_res, out_grad_res, axis.to<int>());
  std::vector<Tensor> x_grad(op_res.size());
  std::transform(op_res.begin(), op_res.end(), x_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return x_grad; 
}

template <>
std::vector<Tensor> concat_grad<LazyTensor>(const std::vector<Tensor>& x, const Tensor& out_grad, const Tensor& axis_) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::concat_grad(x_res, out_grad_res, axis_res);
  std::vector<Tensor> x_grad(op_res.size());
  std::transform(op_res.begin(), op_res.end(), x_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return x_grad; 
}

template <>
Tensor erf_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::erf_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor exp_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::exp_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor expand_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const IntArray& shape) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::expand_grad(x_res, out_grad_res, shape.GetData());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor expand_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& shape_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult shape_res = std::static_pointer_cast<LazyTensor>(shape_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::expand_grad(x_res, out_grad_res, shape_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor gelu_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, bool approximate) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::gelu_grad(x_res, out_grad_res, approximate);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> layer_norm_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& mean, const Tensor& variance, const Tensor& out_grad, float epsilon, int begin_norm_axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value scale_res;
  if(scale) {
    scale_res = std::static_pointer_cast<LazyTensor>(scale.get().impl())->value();
  }
  pir::Value bias_res;
  if(bias) {
    bias_res = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
  }
  pir::Value mean_res = std::static_pointer_cast<LazyTensor>(mean.impl())->value();
  pir::Value variance_res = std::static_pointer_cast<LazyTensor>(variance.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::layer_norm_grad(x_res, scale_res, bias_res, mean_res, variance_res, out_grad_res, epsilon, begin_norm_axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, scale_grad, bias_grad); 
}

template <>
Tensor pow_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Scalar& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::pow_grad(x_res, out_grad_res, y.to<float>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor rsqrt_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::rsqrt_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor silu_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::silu_grad(x_res, out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor square_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::square_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor tanh_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::tanh_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> add_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::add_grad(x_res, y_res, out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
std::tuple<Tensor, Tensor> divide_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::divide_grad(x_res, y_res, out_res, out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor dropout_grad<LazyTensor>(const Tensor& mask, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode) {
  pir::Value mask_res = std::static_pointer_cast<LazyTensor>(mask.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::dropout_grad(mask_res, out_grad_res, p.to<float>(), is_test, mode);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> elementwise_pow_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::elementwise_pow_grad(x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor embedding_grad<LazyTensor>(const Tensor& x, const Tensor& weight, const Tensor& out_grad, int64_t padding_idx, bool sparse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::embedding_grad(x_res, weight_res, out_grad_res, padding_idx, sparse);
  Tensor weight_grad(std::make_shared<LazyTensor>(op_res));
  return weight_grad; 
}

template <>
Tensor fused_softmax_mask_upper_triangle_grad<LazyTensor>(const Tensor& Out, const Tensor& Out_grad) {
  pir::Value Out_res = std::static_pointer_cast<LazyTensor>(Out.impl())->value();
  pir::Value Out_grad_res = std::static_pointer_cast<LazyTensor>(Out_grad.impl())->value();
  auto op_res = paddle::dialect::fused_softmax_mask_upper_triangle_grad(Out_res, Out_grad_res);
  Tensor X_grad(std::make_shared<LazyTensor>(op_res));
  return X_grad; 
}

template <>
std::tuple<Tensor, Tensor> matmul_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x, bool transpose_y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::matmul_grad(x_res, y_res, out_grad_res, transpose_x, transpose_y);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor mean_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::mean_grad(x_res, out_grad_res, axis.GetData(), keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> multiply_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::multiply_grad(x_res, y_res, out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor reshape_grad<LazyTensor>(const Tensor& xshape, const Tensor& out_grad) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::reshape_grad(xshape_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor slice_grad<LazyTensor>(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::slice_grad(input_res, out_grad_res, axes, starts.GetData(), ends.GetData(), infer_flags, decrease_axis);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor slice_grad<LazyTensor>(const Tensor& input, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult starts_res = std::static_pointer_cast<LazyTensor>(starts_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult ends_res = std::static_pointer_cast<LazyTensor>(ends_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::slice_grad(input_res, out_grad_res, starts_res, ends_res, axes, infer_flags, decrease_axis);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor softmax_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad, int axis) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::softmax_grad(out_res, out_grad_res, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor split_grad<LazyTensor>(const std::vector<Tensor>& out_grad, const Scalar& axis) {
  std::vector<pir::Value> out_grad_res(out_grad.size());
  std::transform(out_grad.begin(), out_grad.end(), out_grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::split_grad(out_grad_res, axis.to<int>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor split_grad<LazyTensor>(const std::vector<Tensor>& out_grad, const Tensor& axis_) {
  std::vector<pir::Value> out_grad_res(out_grad.size());
  std::transform(out_grad.begin(), out_grad.end(), out_grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::split_grad(out_grad_res, axis_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor split_with_num_grad<LazyTensor>(const std::vector<Tensor>& out_grad, const Scalar& axis) {
  std::vector<pir::Value> out_grad_res(out_grad.size());
  std::transform(out_grad.begin(), out_grad.end(), out_grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::split_with_num_grad(out_grad_res, axis.to<int>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor split_with_num_grad<LazyTensor>(const std::vector<Tensor>& out_grad, const Tensor& axis_) {
  std::vector<pir::Value> out_grad_res(out_grad.size());
  std::transform(out_grad.begin(), out_grad.end(), out_grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::split_with_num_grad(out_grad_res, axis_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> subtract_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::subtract_grad(x_res, y_res, out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor sum_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sum_grad(x_res, out_grad_res, axis.GetData(), keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor sum_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::sum_grad(x_res, out_grad_res, axis_res, keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor transpose_grad<LazyTensor>(const Tensor& out_grad, const std::vector<int>& perm) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::transpose_grad(out_grad_res, perm);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}


}  // namespace backend
}  // namespace primitive
}  // namespace paddle
