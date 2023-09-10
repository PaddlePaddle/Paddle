// Auto Generated, DO NOT EDIT!

#include "paddle/fluid/primitive/backend/generated/generated_backend.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_api.h"
#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"

namespace paddle {
namespace primitive {
namespace backend {

using LazyTensor = paddle::primitive::LazyTensor;

template <>
Tensor concat<LazyTensor>(const std::vector<Tensor>& x, const Scalar& axis) {
  std::vector<ir::OpResult> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->getValue().dyn_cast<ir::OpResult>();
  });
  auto op_res = paddle::dialect::concat(x_res, axis.to<float>());
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor expand<LazyTensor>(const Tensor& x, const IntArray& shape) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::expand(x_res, shape.GetData());
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor scale<LazyTensor>(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::scale(x_res, scale.to<float>(), bias, bias_after_scale);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor add<LazyTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::add(x_res, y_res);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor add_n<LazyTensor>(const std::vector<Tensor>& inputs) {
  std::vector<ir::OpResult> inputs_res(inputs.size());
  std::transform(inputs.begin(), inputs.end(), inputs_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->getValue().dyn_cast<ir::OpResult>();
  });
  auto op_res = paddle::dialect::add_n(inputs_res);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor divide<LazyTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::divide(x_res, y_res);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor elementwise_pow<LazyTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::elementwise_pow(x_res, y_res);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor full<LazyTensor>(const IntArray& shape, const Scalar& value, DataType dtype, Place place) {
  auto op_res = paddle::dialect::full(shape.GetData(), value.to<float>(), dtype, place);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor mean<LazyTensor>(const Tensor& x, const IntArray& axis, bool keepdim) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::mean(x_res, axis.GetData(), keepdim);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor multiply<LazyTensor>(const Tensor& x, const Tensor& y) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::multiply(x_res, y_res);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
std::tuple<Tensor, Tensor> reshape<LazyTensor>(const Tensor& x, const IntArray& shape) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::reshape(x_res, shape.GetData());
  return std::make_tuple(
    Tensor(std::make_shared<LazyTensor>(std::get<0>(op_res))), 
    Tensor(std::make_shared<LazyTensor>(std::get<1>(op_res)))
  );
 
}

template <>
Tensor sum<LazyTensor>(const Tensor& x, const IntArray& axis, DataType dtype, bool keepdim) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::sum(x_res, axis.GetData(), dtype, keepdim);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor tile<LazyTensor>(const Tensor& x, const IntArray& repeat_times) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::tile(x_res, repeat_times.GetData());
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor tanh_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  ir::OpResult out_res = std::static_pointer_cast<LazyTensor>(out.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::tanh_grad(out_res, out_grad_res);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
std::tuple<Tensor, Tensor> add_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::add_grad(x_res, y_res, out_grad_res, axis);
  return std::make_tuple(
    Tensor(std::make_shared<LazyTensor>(std::get<0>(op_res))), 
    Tensor(std::make_shared<LazyTensor>(std::get<1>(op_res)))
  );
 
}

template <>
std::tuple<Tensor, Tensor> divide_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult y_res = std::static_pointer_cast<LazyTensor>(y.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult out_res = std::static_pointer_cast<LazyTensor>(out.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::divide_grad(x_res, y_res, out_res, out_grad_res, axis);
  return std::make_tuple(
    Tensor(std::make_shared<LazyTensor>(std::get<0>(op_res))), 
    Tensor(std::make_shared<LazyTensor>(std::get<1>(op_res)))
  );
 
}

template <>
Tensor mean_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::mean_grad(x_res, out_grad_res, axis.GetData(), keepdim, reduce_all);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}

template <>
Tensor sum_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all) {
  ir::OpResult x_res = std::static_pointer_cast<LazyTensor>(x.impl())->getValue().dyn_cast<ir::OpResult>();
  ir::OpResult out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->getValue().dyn_cast<ir::OpResult>();
  auto op_res = paddle::dialect::sum_grad(x_res, out_grad_res, axis.GetData(), keepdim, reduce_all);
  return Tensor(std::make_shared<LazyTensor>(op_res));
 
}


}  // namespace backend
}  // namespace primitive
}  // namespace paddle
