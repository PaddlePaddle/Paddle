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
std::tuple<Tensor, Tensor, Tensor> accuracy<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& label) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::accuracy(x_res, indices_res, label_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor accuracy(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor correct(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor total(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(accuracy, correct, total); 
}

template <>
Tensor acos<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::acos(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor acosh<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::acosh(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, const paddle::optional<Tensor>> adagrad_<LazyTensor>(const Tensor& param, const Tensor& grad, const Tensor& moment, const Tensor& learning_rate, const paddle::optional<Tensor>& master_param, float epsilon, bool multi_precision) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value moment_res = std::static_pointer_cast<LazyTensor>(moment.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  auto op_res = paddle::dialect::adagrad_(param_res, grad_res, moment_res, learning_rate_res, master_param_res, epsilon, multi_precision);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor moment_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  paddle::optional<Tensor> master_param_out;
  if(op_res_2){
    master_param_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_2.get())));
  }
  return std::make_tuple(param_out, moment_out, master_param_out); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adam_<LazyTensor>(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  pir::Value moment1_res = std::static_pointer_cast<LazyTensor>(moment1.impl())->value();
  pir::Value moment2_res = std::static_pointer_cast<LazyTensor>(moment2.impl())->value();
  pir::Value beta1_pow_res = std::static_pointer_cast<LazyTensor>(beta1_pow.impl())->value();
  pir::Value beta2_pow_res = std::static_pointer_cast<LazyTensor>(beta2_pow.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  paddle::optional<pir::Value> skip_update_res;
  if(skip_update) {
    pir::Value skip_update_res_inner;
    skip_update_res_inner = std::static_pointer_cast<LazyTensor>(skip_update.get().impl())->value();
    skip_update_res = paddle::make_optional<pir::Value>(skip_update_res_inner);
  }
  auto op_res = paddle::dialect::adam_(param_res, grad_res, learning_rate_res, moment1_res, moment2_res, beta1_pow_res, beta2_pow_res, master_param_res, skip_update_res, beta1.to<float>(), beta2.to<float>(), epsilon.to<float>(), lazy_mode, min_row_size_to_use_multithread, multi_precision, use_global_beta_pow);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor moment1_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor moment2_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor beta1_pow_out(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor beta2_pow_out(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  paddle::optional<Tensor> master_param_out;
  if(op_res_5){
    master_param_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_5.get())));
  }
  return std::make_tuple(param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out, master_param_out); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adam_<LazyTensor>(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Tensor& beta1_, const Tensor& beta2_, const Tensor& epsilon_, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  pir::Value moment1_res = std::static_pointer_cast<LazyTensor>(moment1.impl())->value();
  pir::Value moment2_res = std::static_pointer_cast<LazyTensor>(moment2.impl())->value();
  pir::Value beta1_pow_res = std::static_pointer_cast<LazyTensor>(beta1_pow.impl())->value();
  pir::Value beta2_pow_res = std::static_pointer_cast<LazyTensor>(beta2_pow.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  paddle::optional<pir::Value> skip_update_res;
  if(skip_update) {
    pir::Value skip_update_res_inner;
    skip_update_res_inner = std::static_pointer_cast<LazyTensor>(skip_update.get().impl())->value();
    skip_update_res = paddle::make_optional<pir::Value>(skip_update_res_inner);
  }
  pir::OpResult beta1_res = std::static_pointer_cast<LazyTensor>(beta1_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult beta2_res = std::static_pointer_cast<LazyTensor>(beta2_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult epsilon_res = std::static_pointer_cast<LazyTensor>(epsilon_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::adam_(param_res, grad_res, learning_rate_res, moment1_res, moment2_res, beta1_pow_res, beta2_pow_res, master_param_res, skip_update_res, beta1_res, beta2_res, epsilon_res, lazy_mode, min_row_size_to_use_multithread, multi_precision, use_global_beta_pow);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor moment1_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor moment2_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor beta1_pow_out(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor beta2_pow_out(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  paddle::optional<Tensor> master_param_out;
  if(op_res_5){
    master_param_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_5.get())));
  }
  return std::make_tuple(param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out, master_param_out); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adamax_<LazyTensor>(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment, const Tensor& inf_norm, const Tensor& beta1_pow, const paddle::optional<Tensor>& master_param, float beta1, float beta2, float epsilon, bool multi_precision) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  pir::Value moment_res = std::static_pointer_cast<LazyTensor>(moment.impl())->value();
  pir::Value inf_norm_res = std::static_pointer_cast<LazyTensor>(inf_norm.impl())->value();
  pir::Value beta1_pow_res = std::static_pointer_cast<LazyTensor>(beta1_pow.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  auto op_res = paddle::dialect::adamax_(param_res, grad_res, learning_rate_res, moment_res, inf_norm_res, beta1_pow_res, master_param_res, beta1, beta2, epsilon, multi_precision);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor moment_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor inf_norm_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  paddle::optional<Tensor> master_param_out;
  if(op_res_3){
    master_param_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_3.get())));
  }
  return std::make_tuple(param_out, moment_out, inf_norm_out, master_param_out); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adamw_<LazyTensor>(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, float lr_ratio, float coeff, bool with_decay, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  pir::Value moment1_res = std::static_pointer_cast<LazyTensor>(moment1.impl())->value();
  pir::Value moment2_res = std::static_pointer_cast<LazyTensor>(moment2.impl())->value();
  pir::Value beta1_pow_res = std::static_pointer_cast<LazyTensor>(beta1_pow.impl())->value();
  pir::Value beta2_pow_res = std::static_pointer_cast<LazyTensor>(beta2_pow.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  paddle::optional<pir::Value> skip_update_res;
  if(skip_update) {
    pir::Value skip_update_res_inner;
    skip_update_res_inner = std::static_pointer_cast<LazyTensor>(skip_update.get().impl())->value();
    skip_update_res = paddle::make_optional<pir::Value>(skip_update_res_inner);
  }
  auto op_res = paddle::dialect::adamw_(param_res, grad_res, learning_rate_res, moment1_res, moment2_res, beta1_pow_res, beta2_pow_res, master_param_res, skip_update_res, beta1.to<float>(), beta2.to<float>(), epsilon.to<float>(), lr_ratio, coeff, with_decay, lazy_mode, min_row_size_to_use_multithread, multi_precision, use_global_beta_pow);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor moment1_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor moment2_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor beta1_pow_out(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor beta2_pow_out(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  paddle::optional<Tensor> master_param_out;
  if(op_res_5){
    master_param_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_5.get())));
  }
  return std::make_tuple(param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out, master_param_out); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adamw_<LazyTensor>(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Tensor& beta1_, const Tensor& beta2_, const Tensor& epsilon_, float lr_ratio, float coeff, bool with_decay, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  pir::Value moment1_res = std::static_pointer_cast<LazyTensor>(moment1.impl())->value();
  pir::Value moment2_res = std::static_pointer_cast<LazyTensor>(moment2.impl())->value();
  pir::Value beta1_pow_res = std::static_pointer_cast<LazyTensor>(beta1_pow.impl())->value();
  pir::Value beta2_pow_res = std::static_pointer_cast<LazyTensor>(beta2_pow.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  paddle::optional<pir::Value> skip_update_res;
  if(skip_update) {
    pir::Value skip_update_res_inner;
    skip_update_res_inner = std::static_pointer_cast<LazyTensor>(skip_update.get().impl())->value();
    skip_update_res = paddle::make_optional<pir::Value>(skip_update_res_inner);
  }
  pir::OpResult beta1_res = std::static_pointer_cast<LazyTensor>(beta1_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult beta2_res = std::static_pointer_cast<LazyTensor>(beta2_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult epsilon_res = std::static_pointer_cast<LazyTensor>(epsilon_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::adamw_(param_res, grad_res, learning_rate_res, moment1_res, moment2_res, beta1_pow_res, beta2_pow_res, master_param_res, skip_update_res, beta1_res, beta2_res, epsilon_res, lr_ratio, coeff, with_decay, lazy_mode, min_row_size_to_use_multithread, multi_precision, use_global_beta_pow);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor moment1_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor moment2_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor beta1_pow_out(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor beta2_pow_out(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  paddle::optional<Tensor> master_param_out;
  if(op_res_5){
    master_param_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_5.get())));
  }
  return std::make_tuple(param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out, master_param_out); 
}

template <>
Tensor addmm<LazyTensor>(const Tensor& input, const Tensor& x, const Tensor& y, float beta, float alpha) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::addmm(input_res, x_res, y_res, beta, alpha);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor affine_grid<LazyTensor>(const Tensor& input, const IntArray& output_shape, bool align_corners) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::affine_grid(input_res, output_shape.GetData(), align_corners);
  Tensor output(std::make_shared<LazyTensor>(op_res));
  return output; 
}

template <>
Tensor affine_grid<LazyTensor>(const Tensor& input, const Tensor& output_shape_, bool align_corners) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::OpResult output_shape_res = std::static_pointer_cast<LazyTensor>(output_shape_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::affine_grid(input_res, output_shape_res, align_corners);
  Tensor output(std::make_shared<LazyTensor>(op_res));
  return output; 
}

template <>
Tensor angle<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::angle(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor argmax<LazyTensor>(const Tensor& x, const Scalar& axis, bool keepdims, bool flatten, int dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::argmax(x_res, axis.to<int64_t>(), keepdims, flatten, dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor argmax<LazyTensor>(const Tensor& x, const Tensor& axis_, bool keepdims, bool flatten, int dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::argmax(x_res, axis_res, keepdims, flatten, dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor argmin<LazyTensor>(const Tensor& x, const Scalar& axis, bool keepdims, bool flatten, int dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::argmin(x_res, axis.to<int64_t>(), keepdims, flatten, dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor argmin<LazyTensor>(const Tensor& x, const Tensor& axis_, bool keepdims, bool flatten, int dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::argmin(x_res, axis_res, keepdims, flatten, dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> argsort<LazyTensor>(const Tensor& x, int axis, bool descending) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::argsort(x_res, axis, descending);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor indices(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, indices); 
}

template <>
Tensor as_complex<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::as_complex(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor as_real<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::as_real(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor as_strided<LazyTensor>(const Tensor& input, const std::vector<int64_t>& dims, const std::vector<int64_t>& stride, int64_t offset) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::as_strided(input_res, dims, stride, offset);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor asin<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::asin(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor asinh<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::asinh(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor atan<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::atan(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor atan2<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::atan2(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor atanh<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::atanh(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> auc<LazyTensor>(const Tensor& x, const Tensor& label, const Tensor& stat_pos, const Tensor& stat_neg, const paddle::optional<Tensor>& ins_tag_weight, const std::string& curve, int num_thresholds, int slide_steps) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value stat_pos_res = std::static_pointer_cast<LazyTensor>(stat_pos.impl())->value();
  pir::Value stat_neg_res = std::static_pointer_cast<LazyTensor>(stat_neg.impl())->value();
  paddle::optional<pir::Value> ins_tag_weight_res;
  if(ins_tag_weight) {
    pir::Value ins_tag_weight_res_inner;
    ins_tag_weight_res_inner = std::static_pointer_cast<LazyTensor>(ins_tag_weight.get().impl())->value();
    ins_tag_weight_res = paddle::make_optional<pir::Value>(ins_tag_weight_res_inner);
  }
  auto op_res = paddle::dialect::auc(x_res, label_res, stat_pos_res, stat_neg_res, ins_tag_weight_res, curve, num_thresholds, slide_steps);
  auto op_res_0 = std::get<0>(op_res);
  Tensor auc(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor stat_pos_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor stat_neg_out(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(auc, stat_pos_out, stat_neg_out); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> average_accumulates_<LazyTensor>(const Tensor& param, const Tensor& in_sum_1, const Tensor& in_sum_2, const Tensor& in_sum_3, const Tensor& in_num_accumulates, const Tensor& in_old_num_accumulates, const Tensor& in_num_updates, float average_window, int64_t max_average_window, int64_t min_average_window) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value in_sum_1_res = std::static_pointer_cast<LazyTensor>(in_sum_1.impl())->value();
  pir::Value in_sum_2_res = std::static_pointer_cast<LazyTensor>(in_sum_2.impl())->value();
  pir::Value in_sum_3_res = std::static_pointer_cast<LazyTensor>(in_sum_3.impl())->value();
  pir::Value in_num_accumulates_res = std::static_pointer_cast<LazyTensor>(in_num_accumulates.impl())->value();
  pir::Value in_old_num_accumulates_res = std::static_pointer_cast<LazyTensor>(in_old_num_accumulates.impl())->value();
  pir::Value in_num_updates_res = std::static_pointer_cast<LazyTensor>(in_num_updates.impl())->value();
  auto op_res = paddle::dialect::average_accumulates_(param_res, in_sum_1_res, in_sum_2_res, in_sum_3_res, in_num_accumulates_res, in_old_num_accumulates_res, in_num_updates_res, average_window, max_average_window, min_average_window);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_sum_1(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor out_sum_2(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor out_sum_3(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor out_num_accumulates(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor out_old_num_accumulates(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  Tensor out_num_updates(std::make_shared<LazyTensor>(op_res_5));
  return std::make_tuple(out_sum_1, out_sum_2, out_sum_3, out_num_accumulates, out_old_num_accumulates, out_num_updates); 
}

template <>
Tensor bce_loss<LazyTensor>(const Tensor& input, const Tensor& label) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::bce_loss(input_res, label_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bernoulli<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::bernoulli(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bicubic_interp<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  auto op_res = paddle::dialect::bicubic_interp(x_res, out_size_res, size_tensor_res, scale_tensor_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor output(std::make_shared<LazyTensor>(op_res));
  return output; 
}

template <>
Tensor bilinear<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& weight, const paddle::optional<Tensor>& bias) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  auto op_res = paddle::dialect::bilinear(x_res, y_res, weight_res, bias_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bilinear_interp<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  auto op_res = paddle::dialect::bilinear_interp(x_res, out_size_res, size_tensor_res, scale_tensor_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor output(std::make_shared<LazyTensor>(op_res));
  return output; 
}

template <>
Tensor bincount<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& weights, const Scalar& minlength) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> weights_res;
  if(weights) {
    pir::Value weights_res_inner;
    weights_res_inner = std::static_pointer_cast<LazyTensor>(weights.get().impl())->value();
    weights_res = paddle::make_optional<pir::Value>(weights_res_inner);
  }
  auto op_res = paddle::dialect::bincount(x_res, weights_res, minlength.to<int>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bincount<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& weights, const Tensor& minlength_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> weights_res;
  if(weights) {
    pir::Value weights_res_inner;
    weights_res_inner = std::static_pointer_cast<LazyTensor>(weights.get().impl())->value();
    weights_res = paddle::make_optional<pir::Value>(weights_res_inner);
  }
  pir::OpResult minlength_res = std::static_pointer_cast<LazyTensor>(minlength_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::bincount(x_res, weights_res, minlength_res);
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
Tensor bmm<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::bmm(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor box_coder<LazyTensor>(const Tensor& prior_box, const paddle::optional<Tensor>& prior_box_var, const Tensor& target_box, const std::string& code_type, bool box_normalized, int axis, const std::vector<float>& variance) {
  pir::Value prior_box_res = std::static_pointer_cast<LazyTensor>(prior_box.impl())->value();
  paddle::optional<pir::Value> prior_box_var_res;
  if(prior_box_var) {
    pir::Value prior_box_var_res_inner;
    prior_box_var_res_inner = std::static_pointer_cast<LazyTensor>(prior_box_var.get().impl())->value();
    prior_box_var_res = paddle::make_optional<pir::Value>(prior_box_var_res_inner);
  }
  pir::Value target_box_res = std::static_pointer_cast<LazyTensor>(target_box.impl())->value();
  auto op_res = paddle::dialect::box_coder(prior_box_res, prior_box_var_res, target_box_res, code_type, box_normalized, axis, variance);
  Tensor output_box(std::make_shared<LazyTensor>(op_res));
  return output_box; 
}

template <>
std::vector<Tensor> broadcast_tensors<LazyTensor>(const std::vector<Tensor>& input) {
  std::vector<pir::Value> input_res(input.size());
  std::transform(input.begin(), input.end(), input_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::broadcast_tensors(input_res);
  std::vector<Tensor> out(op_res.size());
  std::transform(op_res.begin(), op_res.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return out; 
}

template <>
Tensor ceil<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::ceil(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor celu<LazyTensor>(const Tensor& x, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::celu(x_res, alpha);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<std::vector<Tensor>, Tensor> check_finite_and_unscale_<LazyTensor>(const std::vector<Tensor>& x, const Tensor& scale) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  auto op_res = paddle::dialect::check_finite_and_unscale_(x_res, scale_res);
  auto op_res_0 = std::get<0>(op_res);
  std::vector<Tensor> out(op_res_0.size());
  std::transform(op_res_0.begin(), op_res_0.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_1 = std::get<1>(op_res);
  Tensor found_infinite(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, found_infinite); 
}

template <>
std::tuple<Tensor, Tensor> check_numerics<LazyTensor>(const Tensor& tensor, const std::string& op_type, const std::string& var_name, int check_nan_inf_level, int stack_height_limit, const std::string& output_dir) {
  pir::Value tensor_res = std::static_pointer_cast<LazyTensor>(tensor.impl())->value();
  auto op_res = paddle::dialect::check_numerics(tensor_res, op_type, var_name, check_nan_inf_level, stack_height_limit, output_dir);
  auto op_res_0 = std::get<0>(op_res);
  Tensor stats(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor values(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(stats, values); 
}

template <>
Tensor cholesky<LazyTensor>(const Tensor& x, bool upper) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cholesky(x_res, upper);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cholesky_solve<LazyTensor>(const Tensor& x, const Tensor& y, bool upper) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::cholesky_solve(x_res, y_res, upper);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> class_center_sample<LazyTensor>(const Tensor& label, int num_classes, int num_samples, int ring_id, int rank, int nranks, bool fix_seed, int seed) {
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::class_center_sample(label_res, num_classes, num_samples, ring_id, rank, nranks, fix_seed, seed);
  auto op_res_0 = std::get<0>(op_res);
  Tensor remapped_label(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor sampled_local_class_center(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(remapped_label, sampled_local_class_center); 
}

template <>
Tensor clip<LazyTensor>(const Tensor& x, const Scalar& min, const Scalar& max) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::clip(x_res, min.to<float>(), max.to<float>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor clip<LazyTensor>(const Tensor& x, const Tensor& min_, const Tensor& max_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult min_res = std::static_pointer_cast<LazyTensor>(min_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult max_res = std::static_pointer_cast<LazyTensor>(max_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::clip(x_res, min_res, max_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor clip_by_norm<LazyTensor>(const Tensor& x, float max_norm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::clip_by_norm(x_res, max_norm);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<std::vector<Tensor>, Tensor> coalesce_tensor<LazyTensor>(const std::vector<Tensor>& input, DataType dtype, bool copy_data, bool set_constant, bool persist_output, float constant, bool use_align, int align_size, int size_of_dtype, const std::vector<int64_t>& concated_shapes, const std::vector<int64_t>& concated_ranks) {
  std::vector<pir::Value> input_res(input.size());
  std::transform(input.begin(), input.end(), input_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::coalesce_tensor(input_res, dtype, copy_data, set_constant, persist_output, constant, use_align, align_size, size_of_dtype, concated_shapes, concated_ranks);
  auto op_res_0 = std::get<0>(op_res);
  std::vector<Tensor> output(op_res_0.size());
  std::transform(op_res_0.begin(), op_res_0.end(), output.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_1 = std::get<1>(op_res);
  Tensor fused_output(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(output, fused_output); 
}

template <>
Tensor complex<LazyTensor>(const Tensor& real, const Tensor& imag) {
  pir::Value real_res = std::static_pointer_cast<LazyTensor>(real.impl())->value();
  pir::Value imag_res = std::static_pointer_cast<LazyTensor>(imag.impl())->value();
  auto op_res = paddle::dialect::complex(real_res, imag_res);
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
Tensor conj<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::conj(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor conv2d<LazyTensor>(const Tensor& input, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  auto op_res = paddle::dialect::conv2d(input_res, filter_res, strides, paddings, padding_algorithm, dilations, groups, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor conv3d<LazyTensor>(const Tensor& input, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  auto op_res = paddle::dialect::conv3d(input_res, filter_res, strides, paddings, padding_algorithm, groups, dilations, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor conv3d_transpose<LazyTensor>(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  auto op_res = paddle::dialect::conv3d_transpose(x_res, filter_res, strides, paddings, output_padding, output_size, padding_algorithm, groups, dilations, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cos<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cos(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cosh<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cosh(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor crop<LazyTensor>(const Tensor& x, const IntArray& shape, const IntArray& offsets) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::crop(x_res, shape.GetData(), offsets.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor crop<LazyTensor>(const Tensor& x, const Tensor& shape_, const Tensor& offsets_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult shape_res = std::static_pointer_cast<LazyTensor>(shape_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult offsets_res = std::static_pointer_cast<LazyTensor>(offsets_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::crop(x_res, shape_res, offsets_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cross<LazyTensor>(const Tensor& x, const Tensor& y, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::cross(x_res, y_res, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> cross_entropy_with_softmax<LazyTensor>(const Tensor& input, const Tensor& label, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::cross_entropy_with_softmax(input_res, label_res, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor softmax(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor loss(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(softmax, loss); 
}

template <>
std::tuple<Tensor, Tensor> cummax<LazyTensor>(const Tensor& x, int axis, int dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cummax(x_res, axis, dtype);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor indices(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, indices); 
}

template <>
std::tuple<Tensor, Tensor> cummin<LazyTensor>(const Tensor& x, int axis, int dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cummin(x_res, axis, dtype);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor indices(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, indices); 
}

template <>
Tensor cumprod<LazyTensor>(const Tensor& x, int dim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cumprod(x_res, dim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cumsum<LazyTensor>(const Tensor& x, const Scalar& axis, bool flatten, bool exclusive, bool reverse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cumsum(x_res, axis.to<int>(), flatten, exclusive, reverse);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cumsum<LazyTensor>(const Tensor& x, const Tensor& axis_, bool flatten, bool exclusive, bool reverse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::cumsum(x_res, axis_res, flatten, exclusive, reverse);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor data<LazyTensor>(const std::string& name, const IntArray& shape, DataType dtype, Place place) {
  auto op_res = paddle::dialect::data(name, shape.GetData(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor depthwise_conv2d<LazyTensor>(const Tensor& input, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  auto op_res = paddle::dialect::depthwise_conv2d(input_res, filter_res, strides, paddings, padding_algorithm, groups, dilations, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor det<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::det(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor diag<LazyTensor>(const Tensor& x, int offset, float padding_value) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::diag(x_res, offset, padding_value);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor diag_embed<LazyTensor>(const Tensor& input, int offset, int dim1, int dim2) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::diag_embed(input_res, offset, dim1, dim2);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor diagonal<LazyTensor>(const Tensor& x, int offset, int axis1, int axis2) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::diagonal(x_res, offset, axis1, axis2);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor digamma<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::digamma(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor dirichlet<LazyTensor>(const Tensor& alpha) {
  pir::Value alpha_res = std::static_pointer_cast<LazyTensor>(alpha.impl())->value();
  auto op_res = paddle::dialect::dirichlet(alpha_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor dist<LazyTensor>(const Tensor& x, const Tensor& y, float p) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::dist(x_res, y_res, p);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor dot<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::dot(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> edit_distance<LazyTensor>(const Tensor& hyps, const Tensor& refs, const paddle::optional<Tensor>& hypslength, const paddle::optional<Tensor>& refslength, bool normalized) {
  pir::Value hyps_res = std::static_pointer_cast<LazyTensor>(hyps.impl())->value();
  pir::Value refs_res = std::static_pointer_cast<LazyTensor>(refs.impl())->value();
  paddle::optional<pir::Value> hypslength_res;
  if(hypslength) {
    pir::Value hypslength_res_inner;
    hypslength_res_inner = std::static_pointer_cast<LazyTensor>(hypslength.get().impl())->value();
    hypslength_res = paddle::make_optional<pir::Value>(hypslength_res_inner);
  }
  paddle::optional<pir::Value> refslength_res;
  if(refslength) {
    pir::Value refslength_res_inner;
    refslength_res_inner = std::static_pointer_cast<LazyTensor>(refslength.get().impl())->value();
    refslength_res = paddle::make_optional<pir::Value>(refslength_res_inner);
  }
  auto op_res = paddle::dialect::edit_distance(hyps_res, refs_res, hypslength_res, refslength_res, normalized);
  auto op_res_0 = std::get<0>(op_res);
  Tensor sequencenum(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(sequencenum, out); 
}

template <>
std::tuple<Tensor, Tensor> eig<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::eig(x_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_w(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor out_v(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_w, out_v); 
}

template <>
std::tuple<Tensor, Tensor> eigh<LazyTensor>(const Tensor& x, const std::string& UPLO) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::eigh(x_res, UPLO);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_w(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor out_v(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_w, out_v); 
}

template <>
Tensor eigvals<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::eigvals(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> eigvalsh<LazyTensor>(const Tensor& x, const std::string& uplo, bool is_test) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::eigvalsh(x_res, uplo, is_test);
  auto op_res_0 = std::get<0>(op_res);
  Tensor eigenvalues(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor eigenvectors(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(eigenvalues, eigenvectors); 
}

template <>
Tensor elu<LazyTensor>(const Tensor& x, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::elu(x_res, alpha);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor equal_all<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::equal_all(x_res, y_res);
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
Tensor erfinv<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::erfinv(x_res);
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
Tensor expand_as<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& y, const std::vector<int>& target_shape) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> y_res;
  if(y) {
    pir::Value y_res_inner;
    y_res_inner = std::static_pointer_cast<LazyTensor>(y.get().impl())->value();
    y_res = paddle::make_optional<pir::Value>(y_res_inner);
  }
  auto op_res = paddle::dialect::expand_as(x_res, y_res, target_shape);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor expm1<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::expm1(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fft_c2c<LazyTensor>(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::fft_c2c(x_res, axes, normalization, forward);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fft_c2r<LazyTensor>(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, int64_t last_dim_size) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::fft_c2r(x_res, axes, normalization, forward, last_dim_size);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fft_r2c<LazyTensor>(const Tensor& x, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, bool onesided) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::fft_r2c(x_res, axes, normalization, forward, onesided);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fill<LazyTensor>(const Tensor& x, const Scalar& value) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::fill(x_res, value.to<float>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fill<LazyTensor>(const Tensor& x, const Tensor& value_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult value_res = std::static_pointer_cast<LazyTensor>(value_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::fill(x_res, value_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fill_diagonal<LazyTensor>(const Tensor& x, float value, int offset, bool wrap) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::fill_diagonal(x_res, value, offset, wrap);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fill_diagonal_tensor<LazyTensor>(const Tensor& x, const Tensor& y, int64_t offset, int dim1, int dim2) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::fill_diagonal_tensor(x_res, y_res, offset, dim1, dim2);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> flash_attn<LazyTensor>(const Tensor& q, const Tensor& k, const Tensor& v, const paddle::optional<Tensor>& fixed_seed_offset, const paddle::optional<Tensor>& attn_mask, float dropout, bool causal, bool return_softmax, bool is_test, const std::string& rng_name) {
  pir::Value q_res = std::static_pointer_cast<LazyTensor>(q.impl())->value();
  pir::Value k_res = std::static_pointer_cast<LazyTensor>(k.impl())->value();
  pir::Value v_res = std::static_pointer_cast<LazyTensor>(v.impl())->value();
  paddle::optional<pir::Value> fixed_seed_offset_res;
  if(fixed_seed_offset) {
    pir::Value fixed_seed_offset_res_inner;
    fixed_seed_offset_res_inner = std::static_pointer_cast<LazyTensor>(fixed_seed_offset.get().impl())->value();
    fixed_seed_offset_res = paddle::make_optional<pir::Value>(fixed_seed_offset_res_inner);
  }
  paddle::optional<pir::Value> attn_mask_res;
  if(attn_mask) {
    pir::Value attn_mask_res_inner;
    attn_mask_res_inner = std::static_pointer_cast<LazyTensor>(attn_mask.get().impl())->value();
    attn_mask_res = paddle::make_optional<pir::Value>(attn_mask_res_inner);
  }
  auto op_res = paddle::dialect::flash_attn(q_res, k_res, v_res, fixed_seed_offset_res, attn_mask_res, dropout, causal, return_softmax, is_test, rng_name);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor softmax(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, softmax); 
}

template <>
std::tuple<Tensor, Tensor> flash_attn_unpadded<LazyTensor>(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const paddle::optional<Tensor>& fixed_seed_offset, const paddle::optional<Tensor>& attn_mask, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout, bool causal, bool return_softmax, bool is_test, const std::string& rng_name) {
  pir::Value q_res = std::static_pointer_cast<LazyTensor>(q.impl())->value();
  pir::Value k_res = std::static_pointer_cast<LazyTensor>(k.impl())->value();
  pir::Value v_res = std::static_pointer_cast<LazyTensor>(v.impl())->value();
  pir::Value cu_seqlens_q_res = std::static_pointer_cast<LazyTensor>(cu_seqlens_q.impl())->value();
  pir::Value cu_seqlens_k_res = std::static_pointer_cast<LazyTensor>(cu_seqlens_k.impl())->value();
  paddle::optional<pir::Value> fixed_seed_offset_res;
  if(fixed_seed_offset) {
    pir::Value fixed_seed_offset_res_inner;
    fixed_seed_offset_res_inner = std::static_pointer_cast<LazyTensor>(fixed_seed_offset.get().impl())->value();
    fixed_seed_offset_res = paddle::make_optional<pir::Value>(fixed_seed_offset_res_inner);
  }
  paddle::optional<pir::Value> attn_mask_res;
  if(attn_mask) {
    pir::Value attn_mask_res_inner;
    attn_mask_res_inner = std::static_pointer_cast<LazyTensor>(attn_mask.get().impl())->value();
    attn_mask_res = paddle::make_optional<pir::Value>(attn_mask_res_inner);
  }
  auto op_res = paddle::dialect::flash_attn_unpadded(q_res, k_res, v_res, cu_seqlens_q_res, cu_seqlens_k_res, fixed_seed_offset_res, attn_mask_res, max_seqlen_q, max_seqlen_k, scale, dropout, causal, return_softmax, is_test, rng_name);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor softmax(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, softmax); 
}

template <>
Tensor flatten<LazyTensor>(const Tensor& x, int start_axis, int stop_axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::flatten(x_res, start_axis, stop_axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor flip<LazyTensor>(const Tensor& x, const std::vector<int>& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::flip(x_res, axis);
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
Tensor fmax<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::fmax(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fmin<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::fmin(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fold<LazyTensor>(const Tensor& x, const std::vector<int>& output_sizes, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::fold(x_res, output_sizes, kernel_sizes, strides, paddings, dilations);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor frame<LazyTensor>(const Tensor& x, int frame_length, int hop_length, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::frame(x_res, frame_length, hop_length, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor full_int_array<LazyTensor>(const IntArray& value, DataType dtype, Place place) {
  auto op_res = paddle::dialect::full_int_array(value.GetData(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor gather<LazyTensor>(const Tensor& x, const Tensor& index, const Scalar& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  auto op_res = paddle::dialect::gather(x_res, index_res, axis.to<int>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor gather<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& axis_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::gather(x_res, index_res, axis_res);
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
Tensor gather_tree<LazyTensor>(const Tensor& ids, const Tensor& parents) {
  pir::Value ids_res = std::static_pointer_cast<LazyTensor>(ids.impl())->value();
  pir::Value parents_res = std::static_pointer_cast<LazyTensor>(parents.impl())->value();
  auto op_res = paddle::dialect::gather_tree(ids_res, parents_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor gaussian_inplace<LazyTensor>(const Tensor& x, float mean, float std, int seed) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::gaussian_inplace(x_res, mean, std, seed);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor gelu<LazyTensor>(const Tensor& x, bool approximate) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::gelu(x_res, approximate);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> generate_proposals<LazyTensor>(const Tensor& scores, const Tensor& bbox_deltas, const Tensor& im_shape, const Tensor& anchors, const Tensor& variances, int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size, float eta, bool pixel_offset) {
  pir::Value scores_res = std::static_pointer_cast<LazyTensor>(scores.impl())->value();
  pir::Value bbox_deltas_res = std::static_pointer_cast<LazyTensor>(bbox_deltas.impl())->value();
  pir::Value im_shape_res = std::static_pointer_cast<LazyTensor>(im_shape.impl())->value();
  pir::Value anchors_res = std::static_pointer_cast<LazyTensor>(anchors.impl())->value();
  pir::Value variances_res = std::static_pointer_cast<LazyTensor>(variances.impl())->value();
  auto op_res = paddle::dialect::generate_proposals(scores_res, bbox_deltas_res, im_shape_res, anchors_res, variances_res, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size, eta, pixel_offset);
  auto op_res_0 = std::get<0>(op_res);
  Tensor rpn_rois(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor rpn_roi_probs(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor rpn_rois_num(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(rpn_rois, rpn_roi_probs, rpn_rois_num); 
}

template <>
Tensor grid_sample<LazyTensor>(const Tensor& x, const Tensor& grid, const std::string& mode, const std::string& padding_mode, bool align_corners) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grid_res = std::static_pointer_cast<LazyTensor>(grid.impl())->value();
  auto op_res = paddle::dialect::grid_sample(x_res, grid_res, mode, padding_mode, align_corners);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor group_norm<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int groups, const std::string& data_layout) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> scale_res;
  if(scale) {
    pir::Value scale_res_inner;
    scale_res_inner = std::static_pointer_cast<LazyTensor>(scale.get().impl())->value();
    scale_res = paddle::make_optional<pir::Value>(scale_res_inner);
  }
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  auto op_res = paddle::dialect::group_norm(x_res, scale_res, bias_res, epsilon, groups, data_layout);
  Tensor y(std::make_shared<LazyTensor>(op_res));
  return y; 
}

template <>
Tensor gumbel_softmax<LazyTensor>(const Tensor& x, float temperature, bool hard, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::gumbel_softmax(x_res, temperature, hard, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor hardshrink<LazyTensor>(const Tensor& x, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::hardshrink(x_res, threshold);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor hardsigmoid<LazyTensor>(const Tensor& x, float slope, float offset) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::hardsigmoid(x_res, slope, offset);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor hardtanh<LazyTensor>(const Tensor& x, float t_min, float t_max) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::hardtanh(x_res, t_min, t_max);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor heaviside<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::heaviside(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor histogram<LazyTensor>(const Tensor& input, int64_t bins, int min, int max) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::histogram(input_res, bins, min, max);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor huber_loss<LazyTensor>(const Tensor& input, const Tensor& label, float delta) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::huber_loss(input_res, label_res, delta);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor i0<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::i0(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor i0e<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::i0e(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor i1<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::i1(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor i1e<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::i1e(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor imag<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::imag(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor index_add<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& add_value, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value add_value_res = std::static_pointer_cast<LazyTensor>(add_value.impl())->value();
  auto op_res = paddle::dialect::index_add(x_res, index_res, add_value_res, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor index_put<LazyTensor>(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, bool accumulate) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  std::vector<pir::Value> indices_res(indices.size());
  std::transform(indices.begin(), indices.end(), indices_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value value_res = std::static_pointer_cast<LazyTensor>(value.impl())->value();
  auto op_res = paddle::dialect::index_put(x_res, indices_res, value_res, accumulate);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor index_sample<LazyTensor>(const Tensor& x, const Tensor& index) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  auto op_res = paddle::dialect::index_sample(x_res, index_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor index_select<LazyTensor>(const Tensor& x, const Tensor& index, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  auto op_res = paddle::dialect::index_select(x_res, index_res, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor index_select_strided<LazyTensor>(const Tensor& x, int64_t index, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::index_select_strided(x_res, index, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor instance_norm<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> scale_res;
  if(scale) {
    pir::Value scale_res_inner;
    scale_res_inner = std::static_pointer_cast<LazyTensor>(scale.get().impl())->value();
    scale_res = paddle::make_optional<pir::Value>(scale_res_inner);
  }
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  auto op_res = paddle::dialect::instance_norm(x_res, scale_res, bias_res, epsilon);
  Tensor y(std::make_shared<LazyTensor>(op_res));
  return y; 
}

template <>
Tensor inverse<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::inverse(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor is_empty<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::is_empty(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor isfinite<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::isfinite(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor isinf<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::isinf(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor isnan<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::isnan(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor kldiv_loss<LazyTensor>(const Tensor& x, const Tensor& label, const std::string& reduction) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::kldiv_loss(x_res, label_res, reduction);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor kron<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::kron(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> kthvalue<LazyTensor>(const Tensor& x, int k, int axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::kthvalue(x_res, k, axis, keepdim);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor indices(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, indices); 
}

template <>
Tensor label_smooth<LazyTensor>(const Tensor& label, const paddle::optional<Tensor>& prior_dist, float epsilon) {
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  paddle::optional<pir::Value> prior_dist_res;
  if(prior_dist) {
    pir::Value prior_dist_res_inner;
    prior_dist_res_inner = std::static_pointer_cast<LazyTensor>(prior_dist.get().impl())->value();
    prior_dist_res = paddle::make_optional<pir::Value>(prior_dist_res_inner);
  }
  auto op_res = paddle::dialect::label_smooth(label_res, prior_dist_res, epsilon);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, const paddle::optional<Tensor>> lamb_<LazyTensor>(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, float weight_decay, float beta1, float beta2, float epsilon, bool always_adapt, bool multi_precision) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  pir::Value moment1_res = std::static_pointer_cast<LazyTensor>(moment1.impl())->value();
  pir::Value moment2_res = std::static_pointer_cast<LazyTensor>(moment2.impl())->value();
  pir::Value beta1_pow_res = std::static_pointer_cast<LazyTensor>(beta1_pow.impl())->value();
  pir::Value beta2_pow_res = std::static_pointer_cast<LazyTensor>(beta2_pow.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  paddle::optional<pir::Value> skip_update_res;
  if(skip_update) {
    pir::Value skip_update_res_inner;
    skip_update_res_inner = std::static_pointer_cast<LazyTensor>(skip_update.get().impl())->value();
    skip_update_res = paddle::make_optional<pir::Value>(skip_update_res_inner);
  }
  auto op_res = paddle::dialect::lamb_(param_res, grad_res, learning_rate_res, moment1_res, moment2_res, beta1_pow_res, beta2_pow_res, master_param_res, skip_update_res, weight_decay, beta1, beta2, epsilon, always_adapt, multi_precision);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor moment1_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor moment2_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor beta1_pow_out(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor beta2_pow_out(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  paddle::optional<Tensor> master_param_outs;
  if(op_res_5){
    master_param_outs = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_5.get())));
  }
  return std::make_tuple(param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out, master_param_outs); 
}

template <>
Tensor layer_norm<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int begin_norm_axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> scale_res;
  if(scale) {
    pir::Value scale_res_inner;
    scale_res_inner = std::static_pointer_cast<LazyTensor>(scale.get().impl())->value();
    scale_res = paddle::make_optional<pir::Value>(scale_res_inner);
  }
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  auto op_res = paddle::dialect::layer_norm(x_res, scale_res, bias_res, epsilon, begin_norm_axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor leaky_relu<LazyTensor>(const Tensor& x, float negative_slope) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::leaky_relu(x_res, negative_slope);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor lerp<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& weight) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  auto op_res = paddle::dialect::lerp(x_res, y_res, weight_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor lgamma<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::lgamma(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor linear_interp<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  auto op_res = paddle::dialect::linear_interp(x_res, out_size_res, size_tensor_res, scale_tensor_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor output(std::make_shared<LazyTensor>(op_res));
  return output; 
}

template <>
Tensor llm_int8_linear<LazyTensor>(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  pir::Value weight_scale_res = std::static_pointer_cast<LazyTensor>(weight_scale.impl())->value();
  auto op_res = paddle::dialect::llm_int8_linear(x_res, weight_res, bias_res, weight_scale_res, threshold);
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
Tensor log10<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::log10(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor log1p<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::log1p(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor log2<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::log2(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor log_loss<LazyTensor>(const Tensor& input, const Tensor& label, float epsilon) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::log_loss(input_res, label_res, epsilon);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor log_softmax<LazyTensor>(const Tensor& x, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::log_softmax(x_res, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logcumsumexp<LazyTensor>(const Tensor& x, int axis, bool flatten, bool exclusive, bool reverse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::logcumsumexp(x_res, axis, flatten, exclusive, reverse);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logical_and<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::logical_and(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logical_not<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::logical_not(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logical_or<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::logical_or(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logical_xor<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::logical_xor(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logit<LazyTensor>(const Tensor& x, float eps) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::logit(x_res, eps);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logsigmoid<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::logsigmoid(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq<LazyTensor>(const Tensor& x, const Tensor& y, const Scalar& rcond, const std::string& driver) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::lstsq(x_res, y_res, rcond.to<float>(), driver);
  auto op_res_0 = std::get<0>(op_res);
  Tensor solution(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor residuals(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor rank(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor singular_values(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(solution, residuals, rank, singular_values); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& rcond_, const std::string& driver) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::OpResult rcond_res = std::static_pointer_cast<LazyTensor>(rcond_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::lstsq(x_res, y_res, rcond_res, driver);
  auto op_res_0 = std::get<0>(op_res);
  Tensor solution(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor residuals(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor rank(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor singular_values(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(solution, residuals, rank, singular_values); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> lu<LazyTensor>(const Tensor& x, bool pivot) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::lu(x_res, pivot);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor pivots(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor infos(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out, pivots, infos); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> lu_unpack<LazyTensor>(const Tensor& x, const Tensor& y, bool unpack_ludata, bool unpack_pivots) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::lu_unpack(x_res, y_res, unpack_ludata, unpack_pivots);
  auto op_res_0 = std::get<0>(op_res);
  Tensor pmat(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor l(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor u(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(pmat, l, u); 
}

template <>
std::tuple<Tensor, Tensor> margin_cross_entropy<LazyTensor>(const Tensor& logits, const Tensor& label, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale) {
  pir::Value logits_res = std::static_pointer_cast<LazyTensor>(logits.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::margin_cross_entropy(logits_res, label_res, return_softmax, ring_id, rank, nranks, margin1, margin2, margin3, scale);
  auto op_res_0 = std::get<0>(op_res);
  Tensor softmax(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor loss(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(softmax, loss); 
}

template <>
std::tuple<Tensor, Tensor, const paddle::optional<Tensor>> masked_multihead_attention_<LazyTensor>(const Tensor& x, const Tensor& cache_kv, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& src_mask, const paddle::optional<Tensor>& cum_offsets, const paddle::optional<Tensor>& sequence_lengths, const paddle::optional<Tensor>& rotary_tensor, const paddle::optional<Tensor>& beam_cache_offset, const paddle::optional<Tensor>& qkv_out_scale, const paddle::optional<Tensor>& out_shift, const paddle::optional<Tensor>& out_smooth, int seq_len, int rotary_emb_dims, bool use_neox_rotary_style, const std::string& compute_dtype, float out_scale, int quant_round_type, float quant_max_bound, float quant_min_bound) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value cache_kv_res = std::static_pointer_cast<LazyTensor>(cache_kv.impl())->value();
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  paddle::optional<pir::Value> src_mask_res;
  if(src_mask) {
    pir::Value src_mask_res_inner;
    src_mask_res_inner = std::static_pointer_cast<LazyTensor>(src_mask.get().impl())->value();
    src_mask_res = paddle::make_optional<pir::Value>(src_mask_res_inner);
  }
  paddle::optional<pir::Value> cum_offsets_res;
  if(cum_offsets) {
    pir::Value cum_offsets_res_inner;
    cum_offsets_res_inner = std::static_pointer_cast<LazyTensor>(cum_offsets.get().impl())->value();
    cum_offsets_res = paddle::make_optional<pir::Value>(cum_offsets_res_inner);
  }
  paddle::optional<pir::Value> sequence_lengths_res;
  if(sequence_lengths) {
    pir::Value sequence_lengths_res_inner;
    sequence_lengths_res_inner = std::static_pointer_cast<LazyTensor>(sequence_lengths.get().impl())->value();
    sequence_lengths_res = paddle::make_optional<pir::Value>(sequence_lengths_res_inner);
  }
  paddle::optional<pir::Value> rotary_tensor_res;
  if(rotary_tensor) {
    pir::Value rotary_tensor_res_inner;
    rotary_tensor_res_inner = std::static_pointer_cast<LazyTensor>(rotary_tensor.get().impl())->value();
    rotary_tensor_res = paddle::make_optional<pir::Value>(rotary_tensor_res_inner);
  }
  paddle::optional<pir::Value> beam_cache_offset_res;
  if(beam_cache_offset) {
    pir::Value beam_cache_offset_res_inner;
    beam_cache_offset_res_inner = std::static_pointer_cast<LazyTensor>(beam_cache_offset.get().impl())->value();
    beam_cache_offset_res = paddle::make_optional<pir::Value>(beam_cache_offset_res_inner);
  }
  paddle::optional<pir::Value> qkv_out_scale_res;
  if(qkv_out_scale) {
    pir::Value qkv_out_scale_res_inner;
    qkv_out_scale_res_inner = std::static_pointer_cast<LazyTensor>(qkv_out_scale.get().impl())->value();
    qkv_out_scale_res = paddle::make_optional<pir::Value>(qkv_out_scale_res_inner);
  }
  paddle::optional<pir::Value> out_shift_res;
  if(out_shift) {
    pir::Value out_shift_res_inner;
    out_shift_res_inner = std::static_pointer_cast<LazyTensor>(out_shift.get().impl())->value();
    out_shift_res = paddle::make_optional<pir::Value>(out_shift_res_inner);
  }
  paddle::optional<pir::Value> out_smooth_res;
  if(out_smooth) {
    pir::Value out_smooth_res_inner;
    out_smooth_res_inner = std::static_pointer_cast<LazyTensor>(out_smooth.get().impl())->value();
    out_smooth_res = paddle::make_optional<pir::Value>(out_smooth_res_inner);
  }
  auto op_res = paddle::dialect::masked_multihead_attention_(x_res, cache_kv_res, bias_res, src_mask_res, cum_offsets_res, sequence_lengths_res, rotary_tensor_res, beam_cache_offset_res, qkv_out_scale_res, out_shift_res, out_smooth_res, seq_len, rotary_emb_dims, use_neox_rotary_style, compute_dtype, out_scale, quant_round_type, quant_max_bound, quant_min_bound);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor cache_kv_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  paddle::optional<Tensor> beam_cache_offset_out;
  if(op_res_2){
    beam_cache_offset_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_2.get())));
  }
  return std::make_tuple(out, cache_kv_out, beam_cache_offset_out); 
}

template <>
Tensor masked_select<LazyTensor>(const Tensor& x, const Tensor& mask) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value mask_res = std::static_pointer_cast<LazyTensor>(mask.impl())->value();
  auto op_res = paddle::dialect::masked_select(x_res, mask_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> matrix_nms<LazyTensor>(const Tensor& bboxes, const Tensor& scores, float score_threshold, int nms_top_k, int keep_top_k, float post_threshold, bool use_gaussian, float gaussian_sigma, int background_label, bool normalized) {
  pir::Value bboxes_res = std::static_pointer_cast<LazyTensor>(bboxes.impl())->value();
  pir::Value scores_res = std::static_pointer_cast<LazyTensor>(scores.impl())->value();
  auto op_res = paddle::dialect::matrix_nms(bboxes_res, scores_res, score_threshold, nms_top_k, keep_top_k, post_threshold, use_gaussian, gaussian_sigma, background_label, normalized);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor index(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor roisnum(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out, index, roisnum); 
}

template <>
Tensor matrix_power<LazyTensor>(const Tensor& x, int n) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::matrix_power(x_res, n);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> max_pool2d_with_index<LazyTensor>(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::max_pool2d_with_index(x_res, kernel_size, strides, paddings, global_pooling, adaptive);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor mask(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, mask); 
}

template <>
std::tuple<Tensor, Tensor> max_pool3d_with_index<LazyTensor>(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::max_pool3d_with_index(x_res, kernel_size, strides, paddings, global_pooling, adaptive);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor mask(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, mask); 
}

template <>
Tensor maxout<LazyTensor>(const Tensor& x, int groups, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::maxout(x_res, groups, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor mean_all<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::mean_all(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> memory_efficient_attention<LazyTensor>(const Tensor& query, const Tensor& key, const Tensor& value, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlens_q, const paddle::optional<Tensor>& cu_seqlens_k, const paddle::optional<Tensor>& causal_diagonal, const paddle::optional<Tensor>& seqlen_k, const Scalar& max_seqlen_q, const Scalar& max_seqlen_k, bool causal, double dropout_p, float scale, bool is_test) {
  pir::Value query_res = std::static_pointer_cast<LazyTensor>(query.impl())->value();
  pir::Value key_res = std::static_pointer_cast<LazyTensor>(key.impl())->value();
  pir::Value value_res = std::static_pointer_cast<LazyTensor>(value.impl())->value();
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  paddle::optional<pir::Value> cu_seqlens_q_res;
  if(cu_seqlens_q) {
    pir::Value cu_seqlens_q_res_inner;
    cu_seqlens_q_res_inner = std::static_pointer_cast<LazyTensor>(cu_seqlens_q.get().impl())->value();
    cu_seqlens_q_res = paddle::make_optional<pir::Value>(cu_seqlens_q_res_inner);
  }
  paddle::optional<pir::Value> cu_seqlens_k_res;
  if(cu_seqlens_k) {
    pir::Value cu_seqlens_k_res_inner;
    cu_seqlens_k_res_inner = std::static_pointer_cast<LazyTensor>(cu_seqlens_k.get().impl())->value();
    cu_seqlens_k_res = paddle::make_optional<pir::Value>(cu_seqlens_k_res_inner);
  }
  paddle::optional<pir::Value> causal_diagonal_res;
  if(causal_diagonal) {
    pir::Value causal_diagonal_res_inner;
    causal_diagonal_res_inner = std::static_pointer_cast<LazyTensor>(causal_diagonal.get().impl())->value();
    causal_diagonal_res = paddle::make_optional<pir::Value>(causal_diagonal_res_inner);
  }
  paddle::optional<pir::Value> seqlen_k_res;
  if(seqlen_k) {
    pir::Value seqlen_k_res_inner;
    seqlen_k_res_inner = std::static_pointer_cast<LazyTensor>(seqlen_k.get().impl())->value();
    seqlen_k_res = paddle::make_optional<pir::Value>(seqlen_k_res_inner);
  }
  auto op_res = paddle::dialect::memory_efficient_attention(query_res, key_res, value_res, bias_res, cu_seqlens_q_res, cu_seqlens_k_res, causal_diagonal_res, seqlen_k_res, max_seqlen_q.to<float>(), max_seqlen_k.to<float>(), causal, dropout_p, scale, is_test);
  auto op_res_0 = std::get<0>(op_res);
  Tensor output(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor logsumexp(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor seed_and_offset(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(output, logsumexp, seed_and_offset); 
}

template <>
Tensor merge_selected_rows<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::merge_selected_rows(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, const paddle::optional<std::vector<Tensor>>> merged_adam_<LazyTensor>(const std::vector<Tensor>& param, const std::vector<Tensor>& grad, const std::vector<Tensor>& learning_rate, const std::vector<Tensor>& moment1, const std::vector<Tensor>& moment2, const std::vector<Tensor>& beta1_pow, const std::vector<Tensor>& beta2_pow, const paddle::optional<std::vector<Tensor>>& master_param, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, bool multi_precision, bool use_global_beta_pow) {
  std::vector<pir::Value> param_res(param.size());
  std::transform(param.begin(), param.end(), param_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> grad_res(grad.size());
  std::transform(grad.begin(), grad.end(), grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> learning_rate_res(learning_rate.size());
  std::transform(learning_rate.begin(), learning_rate.end(), learning_rate_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> moment1_res(moment1.size());
  std::transform(moment1.begin(), moment1.end(), moment1_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> moment2_res(moment2.size());
  std::transform(moment2.begin(), moment2.end(), moment2_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> beta1_pow_res(beta1_pow.size());
  std::transform(beta1_pow.begin(), beta1_pow.end(), beta1_pow_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> beta2_pow_res(beta2_pow.size());
  std::transform(beta2_pow.begin(), beta2_pow.end(), beta2_pow_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  paddle::optional<std::vector<pir::Value>> master_param_res;
  if(master_param) {
    std::vector<pir::Value> master_param_res_inner(master_param.get().size());
    std::transform(master_param.get().begin(), master_param.get().end(), master_param_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    master_param_res = paddle::make_optional<std::vector<pir::Value>>(master_param_res_inner);
  }
  auto op_res = paddle::dialect::merged_adam_(param_res, grad_res, learning_rate_res, moment1_res, moment2_res, beta1_pow_res, beta2_pow_res, master_param_res, beta1.to<float>(), beta2.to<float>(), epsilon.to<float>(), multi_precision, use_global_beta_pow);
  auto op_res_0 = std::get<0>(op_res);
  std::vector<Tensor> param_out(op_res_0.size());
  std::transform(op_res_0.begin(), op_res_0.end(), param_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_1 = std::get<1>(op_res);
  std::vector<Tensor> moment1_out(op_res_1.size());
  std::transform(op_res_1.begin(), op_res_1.end(), moment1_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_2 = std::get<2>(op_res);
  std::vector<Tensor> moment2_out(op_res_2.size());
  std::transform(op_res_2.begin(), op_res_2.end(), moment2_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_3 = std::get<3>(op_res);
  std::vector<Tensor> beta1_pow_out(op_res_3.size());
  std::transform(op_res_3.begin(), op_res_3.end(), beta1_pow_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_4 = std::get<4>(op_res);
  std::vector<Tensor> beta2_pow_out(op_res_4.size());
  std::transform(op_res_4.begin(), op_res_4.end(), beta2_pow_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_5 = std::get<5>(op_res);
  paddle::optional<std::vector<Tensor>> master_param_out;
  if(op_res_5){
    std::vector<Tensor> master_param_out_inner(op_res_5.get().size());
    std::transform(op_res_5.get().begin(), op_res_5.get().end(), master_param_out_inner.begin(), [](const pir::OpResult& res) {
      return Tensor(std::make_shared<LazyTensor>(res));
    });
    master_param_out = paddle::make_optional<std::vector<Tensor>>(master_param_out_inner);
  }
  return std::make_tuple(param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out, master_param_out); 
}

template <>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, const paddle::optional<std::vector<Tensor>>> merged_adam_<LazyTensor>(const std::vector<Tensor>& param, const std::vector<Tensor>& grad, const std::vector<Tensor>& learning_rate, const std::vector<Tensor>& moment1, const std::vector<Tensor>& moment2, const std::vector<Tensor>& beta1_pow, const std::vector<Tensor>& beta2_pow, const paddle::optional<std::vector<Tensor>>& master_param, const Tensor& beta1_, const Tensor& beta2_, const Tensor& epsilon_, bool multi_precision, bool use_global_beta_pow) {
  std::vector<pir::Value> param_res(param.size());
  std::transform(param.begin(), param.end(), param_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> grad_res(grad.size());
  std::transform(grad.begin(), grad.end(), grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> learning_rate_res(learning_rate.size());
  std::transform(learning_rate.begin(), learning_rate.end(), learning_rate_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> moment1_res(moment1.size());
  std::transform(moment1.begin(), moment1.end(), moment1_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> moment2_res(moment2.size());
  std::transform(moment2.begin(), moment2.end(), moment2_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> beta1_pow_res(beta1_pow.size());
  std::transform(beta1_pow.begin(), beta1_pow.end(), beta1_pow_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> beta2_pow_res(beta2_pow.size());
  std::transform(beta2_pow.begin(), beta2_pow.end(), beta2_pow_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  paddle::optional<std::vector<pir::Value>> master_param_res;
  if(master_param) {
    std::vector<pir::Value> master_param_res_inner(master_param.get().size());
    std::transform(master_param.get().begin(), master_param.get().end(), master_param_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    master_param_res = paddle::make_optional<std::vector<pir::Value>>(master_param_res_inner);
  }
  pir::OpResult beta1_res = std::static_pointer_cast<LazyTensor>(beta1_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult beta2_res = std::static_pointer_cast<LazyTensor>(beta2_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult epsilon_res = std::static_pointer_cast<LazyTensor>(epsilon_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::merged_adam_(param_res, grad_res, learning_rate_res, moment1_res, moment2_res, beta1_pow_res, beta2_pow_res, master_param_res, beta1_res, beta2_res, epsilon_res, multi_precision, use_global_beta_pow);
  auto op_res_0 = std::get<0>(op_res);
  std::vector<Tensor> param_out(op_res_0.size());
  std::transform(op_res_0.begin(), op_res_0.end(), param_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_1 = std::get<1>(op_res);
  std::vector<Tensor> moment1_out(op_res_1.size());
  std::transform(op_res_1.begin(), op_res_1.end(), moment1_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_2 = std::get<2>(op_res);
  std::vector<Tensor> moment2_out(op_res_2.size());
  std::transform(op_res_2.begin(), op_res_2.end(), moment2_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_3 = std::get<3>(op_res);
  std::vector<Tensor> beta1_pow_out(op_res_3.size());
  std::transform(op_res_3.begin(), op_res_3.end(), beta1_pow_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_4 = std::get<4>(op_res);
  std::vector<Tensor> beta2_pow_out(op_res_4.size());
  std::transform(op_res_4.begin(), op_res_4.end(), beta2_pow_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_5 = std::get<5>(op_res);
  paddle::optional<std::vector<Tensor>> master_param_out;
  if(op_res_5){
    std::vector<Tensor> master_param_out_inner(op_res_5.get().size());
    std::transform(op_res_5.get().begin(), op_res_5.get().end(), master_param_out_inner.begin(), [](const pir::OpResult& res) {
      return Tensor(std::make_shared<LazyTensor>(res));
    });
    master_param_out = paddle::make_optional<std::vector<Tensor>>(master_param_out_inner);
  }
  return std::make_tuple(param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out, master_param_out); 
}

template <>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, const paddle::optional<std::vector<Tensor>>> merged_momentum_<LazyTensor>(const std::vector<Tensor>& param, const std::vector<Tensor>& grad, const std::vector<Tensor>& velocity, const std::vector<Tensor>& learning_rate, const paddle::optional<std::vector<Tensor>>& master_param, float mu, bool use_nesterov, const std::vector<std::string>& regularization_method, const std::vector<float>& regularization_coeff, bool multi_precision, float rescale_grad) {
  std::vector<pir::Value> param_res(param.size());
  std::transform(param.begin(), param.end(), param_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> grad_res(grad.size());
  std::transform(grad.begin(), grad.end(), grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> velocity_res(velocity.size());
  std::transform(velocity.begin(), velocity.end(), velocity_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> learning_rate_res(learning_rate.size());
  std::transform(learning_rate.begin(), learning_rate.end(), learning_rate_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  paddle::optional<std::vector<pir::Value>> master_param_res;
  if(master_param) {
    std::vector<pir::Value> master_param_res_inner(master_param.get().size());
    std::transform(master_param.get().begin(), master_param.get().end(), master_param_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    master_param_res = paddle::make_optional<std::vector<pir::Value>>(master_param_res_inner);
  }
  auto op_res = paddle::dialect::merged_momentum_(param_res, grad_res, velocity_res, learning_rate_res, master_param_res, mu, use_nesterov, regularization_method, regularization_coeff, multi_precision, rescale_grad);
  auto op_res_0 = std::get<0>(op_res);
  std::vector<Tensor> param_out(op_res_0.size());
  std::transform(op_res_0.begin(), op_res_0.end(), param_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_1 = std::get<1>(op_res);
  std::vector<Tensor> velocity_out(op_res_1.size());
  std::transform(op_res_1.begin(), op_res_1.end(), velocity_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_2 = std::get<2>(op_res);
  paddle::optional<std::vector<Tensor>> master_param_out;
  if(op_res_2){
    std::vector<Tensor> master_param_out_inner(op_res_2.get().size());
    std::transform(op_res_2.get().begin(), op_res_2.get().end(), master_param_out_inner.begin(), [](const pir::OpResult& res) {
      return Tensor(std::make_shared<LazyTensor>(res));
    });
    master_param_out = paddle::make_optional<std::vector<Tensor>>(master_param_out_inner);
  }
  return std::make_tuple(param_out, velocity_out, master_param_out); 
}

template <>
std::vector<Tensor> meshgrid<LazyTensor>(const std::vector<Tensor>& inputs) {
  std::vector<pir::Value> inputs_res(inputs.size());
  std::transform(inputs.begin(), inputs.end(), inputs_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::meshgrid(inputs_res);
  std::vector<Tensor> out(op_res.size());
  std::transform(op_res.begin(), op_res.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return out; 
}

template <>
std::tuple<Tensor, Tensor> mode<LazyTensor>(const Tensor& x, int axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::mode(x_res, axis, keepdim);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor indices(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, indices); 
}

template <>
std::tuple<Tensor, Tensor, const paddle::optional<Tensor>> momentum_<LazyTensor>(const Tensor& param, const Tensor& grad, const Tensor& velocity, const Tensor& learning_rate, const paddle::optional<Tensor>& master_param, float mu, bool use_nesterov, const std::string& regularization_method, float regularization_coeff, bool multi_precision, float rescale_grad) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value velocity_res = std::static_pointer_cast<LazyTensor>(velocity.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  auto op_res = paddle::dialect::momentum_(param_res, grad_res, velocity_res, learning_rate_res, master_param_res, mu, use_nesterov, regularization_method, regularization_coeff, multi_precision, rescale_grad);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor velocity_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  paddle::optional<Tensor> master_param_out;
  if(op_res_2){
    master_param_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_2.get())));
  }
  return std::make_tuple(param_out, velocity_out, master_param_out); 
}

template <>
Tensor multi_dot<LazyTensor>(const std::vector<Tensor>& x) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::multi_dot(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> multiclass_nms3<LazyTensor>(const Tensor& bboxes, const Tensor& scores, const paddle::optional<Tensor>& rois_num, float score_threshold, int nms_top_k, int keep_top_k, float nms_threshold, bool normalized, float nms_eta, int background_label) {
  pir::Value bboxes_res = std::static_pointer_cast<LazyTensor>(bboxes.impl())->value();
  pir::Value scores_res = std::static_pointer_cast<LazyTensor>(scores.impl())->value();
  paddle::optional<pir::Value> rois_num_res;
  if(rois_num) {
    pir::Value rois_num_res_inner;
    rois_num_res_inner = std::static_pointer_cast<LazyTensor>(rois_num.get().impl())->value();
    rois_num_res = paddle::make_optional<pir::Value>(rois_num_res_inner);
  }
  auto op_res = paddle::dialect::multiclass_nms3(bboxes_res, scores_res, rois_num_res, score_threshold, nms_top_k, keep_top_k, nms_threshold, normalized, nms_eta, background_label);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor index(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor nms_rois_num(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out, index, nms_rois_num); 
}

template <>
Tensor multinomial<LazyTensor>(const Tensor& x, const Scalar& num_samples, bool replacement) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::multinomial(x_res, num_samples.to<int>(), replacement);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor multinomial<LazyTensor>(const Tensor& x, const Tensor& num_samples_, bool replacement) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult num_samples_res = std::static_pointer_cast<LazyTensor>(num_samples_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::multinomial(x_res, num_samples_res, replacement);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor multiplex<LazyTensor>(const std::vector<Tensor>& inputs, const Tensor& index) {
  std::vector<pir::Value> inputs_res(inputs.size());
  std::transform(inputs.begin(), inputs.end(), inputs_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  auto op_res = paddle::dialect::multiplex(inputs_res, index_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor mv<LazyTensor>(const Tensor& x, const Tensor& vec) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value vec_res = std::static_pointer_cast<LazyTensor>(vec.impl())->value();
  auto op_res = paddle::dialect::mv(x_res, vec_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor nanmedian<LazyTensor>(const Tensor& x, const IntArray& axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::nanmedian(x_res, axis.GetData(), keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor nearest_interp<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  auto op_res = paddle::dialect::nearest_interp(x_res, out_size_res, size_tensor_res, scale_tensor_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor output(std::make_shared<LazyTensor>(op_res));
  return output; 
}

template <>
Tensor nextafter<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::nextafter(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> nll_loss<LazyTensor>(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, int64_t ignore_index, const std::string& reduction) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  paddle::optional<pir::Value> weight_res;
  if(weight) {
    pir::Value weight_res_inner;
    weight_res_inner = std::static_pointer_cast<LazyTensor>(weight.get().impl())->value();
    weight_res = paddle::make_optional<pir::Value>(weight_res_inner);
  }
  auto op_res = paddle::dialect::nll_loss(input_res, label_res, weight_res, ignore_index, reduction);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor total_weight(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, total_weight); 
}

template <>
Tensor nms<LazyTensor>(const Tensor& x, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::nms(x_res, threshold);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor nonzero<LazyTensor>(const Tensor& condition) {
  pir::Value condition_res = std::static_pointer_cast<LazyTensor>(condition.impl())->value();
  auto op_res = paddle::dialect::nonzero(condition_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor npu_identity<LazyTensor>(const Tensor& x, int format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::npu_identity(x_res, format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor numel<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::numel(x_res);
  Tensor size(std::make_shared<LazyTensor>(op_res));
  return size; 
}

template <>
Tensor overlap_add<LazyTensor>(const Tensor& x, int hop_length, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::overlap_add(x_res, hop_length, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor p_norm<LazyTensor>(const Tensor& x, float porder, int axis, float epsilon, bool keepdim, bool asvector) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::p_norm(x_res, porder, axis, epsilon, keepdim, asvector);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pad3d<LazyTensor>(const Tensor& x, const IntArray& paddings, const std::string& mode, float pad_value, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::pad3d(x_res, paddings.GetData(), mode, pad_value, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pad3d<LazyTensor>(const Tensor& x, const Tensor& paddings_, const std::string& mode, float pad_value, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult paddings_res = std::static_pointer_cast<LazyTensor>(paddings_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::pad3d(x_res, paddings_res, mode, pad_value, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pixel_shuffle<LazyTensor>(const Tensor& x, int upscale_factor, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::pixel_shuffle(x_res, upscale_factor, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pixel_unshuffle<LazyTensor>(const Tensor& x, int downscale_factor, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::pixel_unshuffle(x_res, downscale_factor, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor poisson<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::poisson(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor polygamma<LazyTensor>(const Tensor& x, int n) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::polygamma(x_res, n);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pow<LazyTensor>(const Tensor& x, const Scalar& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::pow(x_res, y.to<float>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor prelu<LazyTensor>(const Tensor& x, const Tensor& alpha, const std::string& data_format, const std::string& mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value alpha_res = std::static_pointer_cast<LazyTensor>(alpha.impl())->value();
  auto op_res = paddle::dialect::prelu(x_res, alpha_res, data_format, mode);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> prior_box<LazyTensor>(const Tensor& input, const Tensor& image, const std::vector<float>& min_sizes, const std::vector<float>& max_sizes, const std::vector<float>& aspect_ratios, const std::vector<float>& variances, bool flip, bool clip, float step_w, float step_h, float offset, bool min_max_aspect_ratios_order) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value image_res = std::static_pointer_cast<LazyTensor>(image.impl())->value();
  auto op_res = paddle::dialect::prior_box(input_res, image_res, min_sizes, max_sizes, aspect_ratios, variances, flip, clip, step_w, step_h, offset, min_max_aspect_ratios_order);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor var(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, var); 
}

template <>
Tensor psroi_pool<LazyTensor>(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, int output_channels, float spatial_scale) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value boxes_res = std::static_pointer_cast<LazyTensor>(boxes.impl())->value();
  paddle::optional<pir::Value> boxes_num_res;
  if(boxes_num) {
    pir::Value boxes_num_res_inner;
    boxes_num_res_inner = std::static_pointer_cast<LazyTensor>(boxes_num.get().impl())->value();
    boxes_num_res = paddle::make_optional<pir::Value>(boxes_num_res_inner);
  }
  auto op_res = paddle::dialect::psroi_pool(x_res, boxes_res, boxes_num_res, pooled_height, pooled_width, output_channels, spatial_scale);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor put_along_axis<LazyTensor>(const Tensor& arr, const Tensor& indices, const Tensor& values, int axis, const std::string& reduce) {
  pir::Value arr_res = std::static_pointer_cast<LazyTensor>(arr.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value values_res = std::static_pointer_cast<LazyTensor>(values.impl())->value();
  auto op_res = paddle::dialect::put_along_axis(arr_res, indices_res, values_res, axis, reduce);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> qr<LazyTensor>(const Tensor& x, const std::string& mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::qr(x_res, mode);
  auto op_res_0 = std::get<0>(op_res);
  Tensor q(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor r(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(q, r); 
}

template <>
Tensor real<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::real(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor reciprocal<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::reciprocal(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> reindex_graph<LazyTensor>(const Tensor& x, const Tensor& neighbors, const Tensor& count, const paddle::optional<Tensor>& hashtable_value, const paddle::optional<Tensor>& hashtable_index) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value neighbors_res = std::static_pointer_cast<LazyTensor>(neighbors.impl())->value();
  pir::Value count_res = std::static_pointer_cast<LazyTensor>(count.impl())->value();
  paddle::optional<pir::Value> hashtable_value_res;
  if(hashtable_value) {
    pir::Value hashtable_value_res_inner;
    hashtable_value_res_inner = std::static_pointer_cast<LazyTensor>(hashtable_value.get().impl())->value();
    hashtable_value_res = paddle::make_optional<pir::Value>(hashtable_value_res_inner);
  }
  paddle::optional<pir::Value> hashtable_index_res;
  if(hashtable_index) {
    pir::Value hashtable_index_res_inner;
    hashtable_index_res_inner = std::static_pointer_cast<LazyTensor>(hashtable_index.get().impl())->value();
    hashtable_index_res = paddle::make_optional<pir::Value>(hashtable_index_res_inner);
  }
  auto op_res = paddle::dialect::reindex_graph(x_res, neighbors_res, count_res, hashtable_value_res, hashtable_index_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor reindex_src(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor reindex_dst(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor out_nodes(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(reindex_src, reindex_dst, out_nodes); 
}

template <>
Tensor relu<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::relu(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor relu6<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::relu6(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor renorm<LazyTensor>(const Tensor& x, float p, int axis, float max_norm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::renorm(x_res, p, axis, max_norm);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor reverse<LazyTensor>(const Tensor& x, const IntArray& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::reverse(x_res, axis.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor reverse<LazyTensor>(const Tensor& x, const Tensor& axis_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::reverse(x_res, axis_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> rms_norm<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& residual, const Tensor& norm_weight, const paddle::optional<Tensor>& norm_bias, float epsilon, int begin_norm_axis, float quant_scale, int quant_round_type, float quant_max_bound, float quant_min_bound) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  paddle::optional<pir::Value> residual_res;
  if(residual) {
    pir::Value residual_res_inner;
    residual_res_inner = std::static_pointer_cast<LazyTensor>(residual.get().impl())->value();
    residual_res = paddle::make_optional<pir::Value>(residual_res_inner);
  }
  pir::Value norm_weight_res = std::static_pointer_cast<LazyTensor>(norm_weight.impl())->value();
  paddle::optional<pir::Value> norm_bias_res;
  if(norm_bias) {
    pir::Value norm_bias_res_inner;
    norm_bias_res_inner = std::static_pointer_cast<LazyTensor>(norm_bias.get().impl())->value();
    norm_bias_res = paddle::make_optional<pir::Value>(norm_bias_res_inner);
  }
  auto op_res = paddle::dialect::rms_norm(x_res, bias_res, residual_res, norm_weight_res, norm_bias_res, epsilon, begin_norm_axis, quant_scale, quant_round_type, quant_max_bound, quant_min_bound);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor residual_out(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, residual_out); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, const paddle::optional<Tensor>, const paddle::optional<Tensor>> rmsprop_<LazyTensor>(const Tensor& param, const Tensor& mean_square, const Tensor& grad, const Tensor& moment, const Tensor& learning_rate, const paddle::optional<Tensor>& mean_grad, const paddle::optional<Tensor>& master_param, float epsilon, float decay, float momentum, bool centered, bool multi_precision) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value mean_square_res = std::static_pointer_cast<LazyTensor>(mean_square.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value moment_res = std::static_pointer_cast<LazyTensor>(moment.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  paddle::optional<pir::Value> mean_grad_res;
  if(mean_grad) {
    pir::Value mean_grad_res_inner;
    mean_grad_res_inner = std::static_pointer_cast<LazyTensor>(mean_grad.get().impl())->value();
    mean_grad_res = paddle::make_optional<pir::Value>(mean_grad_res_inner);
  }
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  auto op_res = paddle::dialect::rmsprop_(param_res, mean_square_res, grad_res, moment_res, learning_rate_res, mean_grad_res, master_param_res, epsilon, decay, momentum, centered, multi_precision);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor moment_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor mean_square_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  paddle::optional<Tensor> mean_grad_out;
  if(op_res_3){
    mean_grad_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_3.get())));
  }
  auto op_res_4 = std::get<4>(op_res);
  paddle::optional<Tensor> master_param_outs;
  if(op_res_4){
    master_param_outs = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_4.get())));
  }
  return std::make_tuple(param_out, moment_out, mean_square_out, mean_grad_out, master_param_outs); 
}

template <>
Tensor roi_align<LazyTensor>(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value boxes_res = std::static_pointer_cast<LazyTensor>(boxes.impl())->value();
  paddle::optional<pir::Value> boxes_num_res;
  if(boxes_num) {
    pir::Value boxes_num_res_inner;
    boxes_num_res_inner = std::static_pointer_cast<LazyTensor>(boxes_num.get().impl())->value();
    boxes_num_res = paddle::make_optional<pir::Value>(boxes_num_res_inner);
  }
  auto op_res = paddle::dialect::roi_align(x_res, boxes_res, boxes_num_res, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor roi_pool<LazyTensor>(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, float spatial_scale) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value boxes_res = std::static_pointer_cast<LazyTensor>(boxes.impl())->value();
  paddle::optional<pir::Value> boxes_num_res;
  if(boxes_num) {
    pir::Value boxes_num_res_inner;
    boxes_num_res_inner = std::static_pointer_cast<LazyTensor>(boxes_num.get().impl())->value();
    boxes_num_res = paddle::make_optional<pir::Value>(boxes_num_res_inner);
  }
  auto op_res = paddle::dialect::roi_pool(x_res, boxes_res, boxes_num_res, pooled_height, pooled_width, spatial_scale);
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
Tensor round<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::round(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor rsqrt<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::rsqrt(x_res);
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
Tensor searchsorted<LazyTensor>(const Tensor& sorted_sequence, const Tensor& values, bool out_int32, bool right) {
  pir::Value sorted_sequence_res = std::static_pointer_cast<LazyTensor>(sorted_sequence.impl())->value();
  pir::Value values_res = std::static_pointer_cast<LazyTensor>(values.impl())->value();
  auto op_res = paddle::dialect::searchsorted(sorted_sequence_res, values_res, out_int32, right);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor segment_pool<LazyTensor>(const Tensor& x, const Tensor& segment_ids, const std::string& pooltype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value segment_ids_res = std::static_pointer_cast<LazyTensor>(segment_ids.impl())->value();
  auto op_res = paddle::dialect::segment_pool(x_res, segment_ids_res, pooltype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor selu<LazyTensor>(const Tensor& x, float scale, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::selu(x_res, scale, alpha);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor send_u_recv<LazyTensor>(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& reduce_op, const IntArray& out_size) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value src_index_res = std::static_pointer_cast<LazyTensor>(src_index.impl())->value();
  pir::Value dst_index_res = std::static_pointer_cast<LazyTensor>(dst_index.impl())->value();
  auto op_res = paddle::dialect::send_u_recv(x_res, src_index_res, dst_index_res, reduce_op, out_size.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor send_u_recv<LazyTensor>(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const Tensor& out_size_, const std::string& reduce_op) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value src_index_res = std::static_pointer_cast<LazyTensor>(src_index.impl())->value();
  pir::Value dst_index_res = std::static_pointer_cast<LazyTensor>(dst_index.impl())->value();
  pir::OpResult out_size_res = std::static_pointer_cast<LazyTensor>(out_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::send_u_recv(x_res, src_index_res, dst_index_res, out_size_res, reduce_op);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor send_ue_recv<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op, const std::string& reduce_op, const IntArray& out_size) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value src_index_res = std::static_pointer_cast<LazyTensor>(src_index.impl())->value();
  pir::Value dst_index_res = std::static_pointer_cast<LazyTensor>(dst_index.impl())->value();
  auto op_res = paddle::dialect::send_ue_recv(x_res, y_res, src_index_res, dst_index_res, message_op, reduce_op, out_size.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor send_ue_recv<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const Tensor& out_size_, const std::string& message_op, const std::string& reduce_op) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value src_index_res = std::static_pointer_cast<LazyTensor>(src_index.impl())->value();
  pir::Value dst_index_res = std::static_pointer_cast<LazyTensor>(dst_index.impl())->value();
  pir::OpResult out_size_res = std::static_pointer_cast<LazyTensor>(out_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::send_ue_recv(x_res, y_res, src_index_res, dst_index_res, out_size_res, message_op, reduce_op);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor send_uv<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const std::string& message_op) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value src_index_res = std::static_pointer_cast<LazyTensor>(src_index.impl())->value();
  pir::Value dst_index_res = std::static_pointer_cast<LazyTensor>(dst_index.impl())->value();
  auto op_res = paddle::dialect::send_uv(x_res, y_res, src_index_res, dst_index_res, message_op);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, const paddle::optional<Tensor>> sgd_<LazyTensor>(const Tensor& param, const Tensor& learning_rate, const Tensor& grad, const paddle::optional<Tensor>& master_param, bool multi_precision) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  auto op_res = paddle::dialect::sgd_(param_res, learning_rate_res, grad_res, master_param_res, multi_precision);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  paddle::optional<Tensor> master_param_out;
  if(op_res_1){
    master_param_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_1.get())));
  }
  return std::make_tuple(param_out, master_param_out); 
}

template <>
Tensor shadow_output<LazyTensor>(const Tensor& x, const std::string& name) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::shadow_output(x_res, name);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor shape<LazyTensor>(const Tensor& input) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::shape(input_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor shard_index<LazyTensor>(const Tensor& input, int index_num, int nshards, int shard_id, int ignore_value) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::shard_index(input_res, index_num, nshards, shard_id, ignore_value);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sigmoid<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sigmoid(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sigmoid_cross_entropy_with_logits<LazyTensor>(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, bool normalize, int ignore_index) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  paddle::optional<pir::Value> pos_weight_res;
  if(pos_weight) {
    pir::Value pos_weight_res_inner;
    pos_weight_res_inner = std::static_pointer_cast<LazyTensor>(pos_weight.get().impl())->value();
    pos_weight_res = paddle::make_optional<pir::Value>(pos_weight_res_inner);
  }
  auto op_res = paddle::dialect::sigmoid_cross_entropy_with_logits(x_res, label_res, pos_weight_res, normalize, ignore_index);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sign<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sign(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor silu<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::silu(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sin<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sin(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sinh<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sinh(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor slogdet<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::slogdet(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor softplus<LazyTensor>(const Tensor& x, float beta, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::softplus(x_res, beta, threshold);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor softshrink<LazyTensor>(const Tensor& x, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::softshrink(x_res, threshold);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor softsign<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::softsign(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor solve<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::solve(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor spectral_norm<LazyTensor>(const Tensor& weight, const Tensor& u, const Tensor& v, int dim, int power_iters, float eps) {
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  pir::Value u_res = std::static_pointer_cast<LazyTensor>(u.impl())->value();
  pir::Value v_res = std::static_pointer_cast<LazyTensor>(v.impl())->value();
  auto op_res = paddle::dialect::spectral_norm(weight_res, u_res, v_res, dim, power_iters, eps);
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
Tensor square<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::square(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor squared_l2_norm<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::squared_l2_norm(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor squeeze<LazyTensor>(const Tensor& x, const IntArray& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::squeeze(x_res, axis.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor squeeze<LazyTensor>(const Tensor& x, const Tensor& axis_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::squeeze(x_res, axis_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor stack<LazyTensor>(const std::vector<Tensor>& x, int axis) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::stack(x_res, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor stanh<LazyTensor>(const Tensor& x, float scale_a, float scale_b) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::stanh(x_res, scale_a, scale_b);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> svd<LazyTensor>(const Tensor& x, bool full_matrices) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::svd(x_res, full_matrices);
  auto op_res_0 = std::get<0>(op_res);
  Tensor u(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor s(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor vh(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(u, s, vh); 
}

template <>
Tensor take_along_axis<LazyTensor>(const Tensor& arr, const Tensor& indices, int axis) {
  pir::Value arr_res = std::static_pointer_cast<LazyTensor>(arr.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  auto op_res = paddle::dialect::take_along_axis(arr_res, indices_res, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor tan<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::tan(x_res);
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
Tensor tanh_shrink<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::tanh_shrink(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor temporal_shift<LazyTensor>(const Tensor& x, int seg_num, float shift_ratio, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::temporal_shift(x_res, seg_num, shift_ratio, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor tensor_unfold<LazyTensor>(const Tensor& input, int64_t axis, int64_t size, int64_t step) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::tensor_unfold(input_res, axis, size, step);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor thresholded_relu<LazyTensor>(const Tensor& x, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::thresholded_relu(x_res, threshold);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> topk<LazyTensor>(const Tensor& x, const Scalar& k, int axis, bool largest, bool sorted) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::topk(x_res, k.to<int>(), axis, largest, sorted);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor indices(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, indices); 
}

template <>
std::tuple<Tensor, Tensor> topk<LazyTensor>(const Tensor& x, const Tensor& k_, int axis, bool largest, bool sorted) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult k_res = std::static_pointer_cast<LazyTensor>(k_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::topk(x_res, k_res, axis, largest, sorted);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor indices(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, indices); 
}

template <>
Tensor trace<LazyTensor>(const Tensor& x, int offset, int axis1, int axis2) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::trace(x_res, offset, axis1, axis2);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor triangular_solve<LazyTensor>(const Tensor& x, const Tensor& y, bool upper, bool transpose, bool unitriangular) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::triangular_solve(x_res, y_res, upper, transpose, unitriangular);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor trilinear_interp<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  auto op_res = paddle::dialect::trilinear_interp(x_res, out_size_res, size_tensor_res, scale_tensor_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor output(std::make_shared<LazyTensor>(op_res));
  return output; 
}

template <>
Tensor trunc<LazyTensor>(const Tensor& input) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::trunc(input_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::vector<Tensor> unbind<LazyTensor>(const Tensor& input, int axis) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::unbind(input_res, axis);
  std::vector<Tensor> out(op_res.size());
  std::transform(op_res.begin(), op_res.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return out; 
}

template <>
Tensor unfold<LazyTensor>(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::unfold(x_res, kernel_sizes, strides, paddings, dilations);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor uniform_inplace<LazyTensor>(const Tensor& x, float min, float max, int seed, int diag_num, int diag_step, float diag_val) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::uniform_inplace(x_res, min, max, seed, diag_num, diag_step, diag_val);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> unique_consecutive<LazyTensor>(const Tensor& x, bool return_inverse, bool return_counts, const std::vector<int>& axis, int dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::unique_consecutive(x_res, return_inverse, return_counts, axis, dtype);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor index(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor counts(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out, index, counts); 
}

template <>
Tensor unpool3d<LazyTensor>(const Tensor& x, const Tensor& indices, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_size, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  auto op_res = paddle::dialect::unpool3d(x_res, indices_res, ksize, strides, paddings, output_size, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor unsqueeze<LazyTensor>(const Tensor& x, const IntArray& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::unsqueeze(x_res, axis.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor unsqueeze<LazyTensor>(const Tensor& x, const Tensor& axis_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::unsqueeze(x_res, axis_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::vector<Tensor> unstack<LazyTensor>(const Tensor& x, int axis, int num) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::unstack(x_res, axis, num);
  std::vector<Tensor> out(op_res.size());
  std::transform(op_res.begin(), op_res.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return out; 
}

template <>
std::tuple<std::vector<Tensor>, Tensor, Tensor, Tensor> update_loss_scaling_<LazyTensor>(const std::vector<Tensor>& x, const Tensor& found_infinite, const Tensor& prev_loss_scaling, const Tensor& in_good_steps, const Tensor& in_bad_steps, int incr_every_n_steps, int decr_every_n_nan_or_inf, float incr_ratio, float decr_ratio, const Scalar& stop_update) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value found_infinite_res = std::static_pointer_cast<LazyTensor>(found_infinite.impl())->value();
  pir::Value prev_loss_scaling_res = std::static_pointer_cast<LazyTensor>(prev_loss_scaling.impl())->value();
  pir::Value in_good_steps_res = std::static_pointer_cast<LazyTensor>(in_good_steps.impl())->value();
  pir::Value in_bad_steps_res = std::static_pointer_cast<LazyTensor>(in_bad_steps.impl())->value();
  auto op_res = paddle::dialect::update_loss_scaling_(x_res, found_infinite_res, prev_loss_scaling_res, in_good_steps_res, in_bad_steps_res, incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio, stop_update.to<bool>());
  auto op_res_0 = std::get<0>(op_res);
  std::vector<Tensor> out(op_res_0.size());
  std::transform(op_res_0.begin(), op_res_0.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_1 = std::get<1>(op_res);
  Tensor loss_scaling(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor out_good_steps(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor out_bad_steps(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(out, loss_scaling, out_good_steps, out_bad_steps); 
}

template <>
std::tuple<std::vector<Tensor>, Tensor, Tensor, Tensor> update_loss_scaling_<LazyTensor>(const std::vector<Tensor>& x, const Tensor& found_infinite, const Tensor& prev_loss_scaling, const Tensor& in_good_steps, const Tensor& in_bad_steps, const Tensor& stop_update_, int incr_every_n_steps, int decr_every_n_nan_or_inf, float incr_ratio, float decr_ratio) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value found_infinite_res = std::static_pointer_cast<LazyTensor>(found_infinite.impl())->value();
  pir::Value prev_loss_scaling_res = std::static_pointer_cast<LazyTensor>(prev_loss_scaling.impl())->value();
  pir::Value in_good_steps_res = std::static_pointer_cast<LazyTensor>(in_good_steps.impl())->value();
  pir::Value in_bad_steps_res = std::static_pointer_cast<LazyTensor>(in_bad_steps.impl())->value();
  pir::OpResult stop_update_res = std::static_pointer_cast<LazyTensor>(stop_update_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::update_loss_scaling_(x_res, found_infinite_res, prev_loss_scaling_res, in_good_steps_res, in_bad_steps_res, stop_update_res, incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio);
  auto op_res_0 = std::get<0>(op_res);
  std::vector<Tensor> out(op_res_0.size());
  std::transform(op_res_0.begin(), op_res_0.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_1 = std::get<1>(op_res);
  Tensor loss_scaling(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor out_good_steps(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor out_bad_steps(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(out, loss_scaling, out_good_steps, out_bad_steps); 
}

template <>
Tensor variable_length_memory_efficient_attention<LazyTensor>(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& seq_lens, const Tensor& kv_seq_lens, const paddle::optional<Tensor>& mask, float scale, bool causal) {
  pir::Value query_res = std::static_pointer_cast<LazyTensor>(query.impl())->value();
  pir::Value key_res = std::static_pointer_cast<LazyTensor>(key.impl())->value();
  pir::Value value_res = std::static_pointer_cast<LazyTensor>(value.impl())->value();
  pir::Value seq_lens_res = std::static_pointer_cast<LazyTensor>(seq_lens.impl())->value();
  pir::Value kv_seq_lens_res = std::static_pointer_cast<LazyTensor>(kv_seq_lens.impl())->value();
  paddle::optional<pir::Value> mask_res;
  if(mask) {
    pir::Value mask_res_inner;
    mask_res_inner = std::static_pointer_cast<LazyTensor>(mask.get().impl())->value();
    mask_res = paddle::make_optional<pir::Value>(mask_res_inner);
  }
  auto op_res = paddle::dialect::variable_length_memory_efficient_attention(query_res, key_res, value_res, seq_lens_res, kv_seq_lens_res, mask_res, scale, causal);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor view_dtype<LazyTensor>(const Tensor& input, DataType dtype) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::view_dtype(input_res, dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor view_shape<LazyTensor>(const Tensor& input, const std::vector<int64_t>& dims) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::view_shape(input_res, dims);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> viterbi_decode<LazyTensor>(const Tensor& potentials, const Tensor& transition_params, const Tensor& lengths, bool include_bos_eos_tag) {
  pir::Value potentials_res = std::static_pointer_cast<LazyTensor>(potentials.impl())->value();
  pir::Value transition_params_res = std::static_pointer_cast<LazyTensor>(transition_params.impl())->value();
  pir::Value lengths_res = std::static_pointer_cast<LazyTensor>(lengths.impl())->value();
  auto op_res = paddle::dialect::viterbi_decode(potentials_res, transition_params_res, lengths_res, include_bos_eos_tag);
  auto op_res_0 = std::get<0>(op_res);
  Tensor scores(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor path(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(scores, path); 
}

template <>
Tensor warpctc<LazyTensor>(const Tensor& logits, const Tensor& label, const paddle::optional<Tensor>& logits_length, const paddle::optional<Tensor>& labels_length, int blank, bool norm_by_times) {
  pir::Value logits_res = std::static_pointer_cast<LazyTensor>(logits.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  paddle::optional<pir::Value> logits_length_res;
  if(logits_length) {
    pir::Value logits_length_res_inner;
    logits_length_res_inner = std::static_pointer_cast<LazyTensor>(logits_length.get().impl())->value();
    logits_length_res = paddle::make_optional<pir::Value>(logits_length_res_inner);
  }
  paddle::optional<pir::Value> labels_length_res;
  if(labels_length) {
    pir::Value labels_length_res_inner;
    labels_length_res_inner = std::static_pointer_cast<LazyTensor>(labels_length.get().impl())->value();
    labels_length_res = paddle::make_optional<pir::Value>(labels_length_res_inner);
  }
  auto op_res = paddle::dialect::warpctc(logits_res, label_res, logits_length_res, labels_length_res, blank, norm_by_times);
  Tensor loss(std::make_shared<LazyTensor>(op_res));
  return loss; 
}

template <>
Tensor warprnnt<LazyTensor>(const Tensor& input, const Tensor& label, const Tensor& input_lengths, const Tensor& label_lengths, int blank, float fastemit_lambda) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value input_lengths_res = std::static_pointer_cast<LazyTensor>(input_lengths.impl())->value();
  pir::Value label_lengths_res = std::static_pointer_cast<LazyTensor>(label_lengths.impl())->value();
  auto op_res = paddle::dialect::warprnnt(input_res, label_res, input_lengths_res, label_lengths_res, blank, fastemit_lambda);
  Tensor loss(std::make_shared<LazyTensor>(op_res));
  return loss; 
}

template <>
Tensor weight_dequantize<LazyTensor>(const Tensor& x, const Tensor& scale, const std::string& algo, DataType out_dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  auto op_res = paddle::dialect::weight_dequantize(x_res, scale_res, algo, out_dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor weight_only_linear<LazyTensor>(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, const std::string& weight_dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  pir::Value weight_scale_res = std::static_pointer_cast<LazyTensor>(weight_scale.impl())->value();
  auto op_res = paddle::dialect::weight_only_linear(x_res, weight_res, bias_res, weight_scale_res, weight_dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> weight_quantize<LazyTensor>(const Tensor& x, const std::string& algo) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::weight_quantize(x_res, algo);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, scale); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> weighted_sample_neighbors<LazyTensor>(const Tensor& row, const Tensor& colptr, const Tensor& edge_weight, const Tensor& input_nodes, const paddle::optional<Tensor>& eids, int sample_size, bool return_eids) {
  pir::Value row_res = std::static_pointer_cast<LazyTensor>(row.impl())->value();
  pir::Value colptr_res = std::static_pointer_cast<LazyTensor>(colptr.impl())->value();
  pir::Value edge_weight_res = std::static_pointer_cast<LazyTensor>(edge_weight.impl())->value();
  pir::Value input_nodes_res = std::static_pointer_cast<LazyTensor>(input_nodes.impl())->value();
  paddle::optional<pir::Value> eids_res;
  if(eids) {
    pir::Value eids_res_inner;
    eids_res_inner = std::static_pointer_cast<LazyTensor>(eids.get().impl())->value();
    eids_res = paddle::make_optional<pir::Value>(eids_res_inner);
  }
  auto op_res = paddle::dialect::weighted_sample_neighbors(row_res, colptr_res, edge_weight_res, input_nodes_res, eids_res, sample_size, return_eids);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_neighbors(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor out_count(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor out_eids(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out_neighbors, out_count, out_eids); 
}

template <>
Tensor where<LazyTensor>(const Tensor& condition, const Tensor& x, const Tensor& y) {
  pir::Value condition_res = std::static_pointer_cast<LazyTensor>(condition.impl())->value();
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::where(condition_res, x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> yolo_box<LazyTensor>(const Tensor& x, const Tensor& img_size, const std::vector<int>& anchors, int class_num, float conf_thresh, int downsample_ratio, bool clip_bbox, float scale_x_y, bool iou_aware, float iou_aware_factor) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value img_size_res = std::static_pointer_cast<LazyTensor>(img_size.impl())->value();
  auto op_res = paddle::dialect::yolo_box(x_res, img_size_res, anchors, class_num, conf_thresh, downsample_ratio, clip_bbox, scale_x_y, iou_aware, iou_aware_factor);
  auto op_res_0 = std::get<0>(op_res);
  Tensor boxes(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scores(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(boxes, scores); 
}

template <>
Tensor yolo_loss<LazyTensor>(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const std::vector<int>& anchors, const std::vector<int>& anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth, float scale_x_y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value gt_box_res = std::static_pointer_cast<LazyTensor>(gt_box.impl())->value();
  pir::Value gt_label_res = std::static_pointer_cast<LazyTensor>(gt_label.impl())->value();
  paddle::optional<pir::Value> gt_score_res;
  if(gt_score) {
    pir::Value gt_score_res_inner;
    gt_score_res_inner = std::static_pointer_cast<LazyTensor>(gt_score.get().impl())->value();
    gt_score_res = paddle::make_optional<pir::Value>(gt_score_res_inner);
  }
  auto op_res = paddle::dialect::yolo_loss(x_res, gt_box_res, gt_label_res, gt_score_res, anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, use_label_smooth, scale_x_y);
  Tensor loss(std::make_shared<LazyTensor>(op_res));
  return loss; 
}

template <>
std::tuple<Tensor, Tensor, Tensor, const paddle::optional<Tensor>> adadelta_<LazyTensor>(const Tensor& param, const Tensor& grad, const Tensor& avg_squared_grad, const Tensor& avg_squared_update, const Tensor& learning_rate, const paddle::optional<Tensor>& master_param, float rho, float epsilon, bool multi_precision) {
  pir::Value param_res = std::static_pointer_cast<LazyTensor>(param.impl())->value();
  pir::Value grad_res = std::static_pointer_cast<LazyTensor>(grad.impl())->value();
  pir::Value avg_squared_grad_res = std::static_pointer_cast<LazyTensor>(avg_squared_grad.impl())->value();
  pir::Value avg_squared_update_res = std::static_pointer_cast<LazyTensor>(avg_squared_update.impl())->value();
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  paddle::optional<pir::Value> master_param_res;
  if(master_param) {
    pir::Value master_param_res_inner;
    master_param_res_inner = std::static_pointer_cast<LazyTensor>(master_param.get().impl())->value();
    master_param_res = paddle::make_optional<pir::Value>(master_param_res_inner);
  }
  auto op_res = paddle::dialect::adadelta_(param_res, grad_res, avg_squared_grad_res, avg_squared_update_res, learning_rate_res, master_param_res, rho, epsilon, multi_precision);
  auto op_res_0 = std::get<0>(op_res);
  Tensor param_out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor moment_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor inf_norm_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  paddle::optional<Tensor> master_param_out;
  if(op_res_3){
    master_param_out = paddle::make_optional<Tensor>(Tensor(std::make_shared<LazyTensor>(op_res_3.get())));
  }
  return std::make_tuple(param_out, moment_out, inf_norm_out, master_param_out); 
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
Tensor all<LazyTensor>(const Tensor& x, const std::vector<int64_t>& axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::all(x_res, axis, keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor amax<LazyTensor>(const Tensor& x, const std::vector<int64_t>& axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::amax(x_res, axis, keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor amin<LazyTensor>(const Tensor& x, const std::vector<int64_t>& axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::amin(x_res, axis, keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor any<LazyTensor>(const Tensor& x, const std::vector<int64_t>& axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::any(x_res, axis, keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor arange<LazyTensor>(const Tensor& start, const Tensor& end, const Tensor& step, DataType dtype, Place place) {
  pir::Value start_res = std::static_pointer_cast<LazyTensor>(start.impl())->value();
  pir::Value end_res = std::static_pointer_cast<LazyTensor>(end.impl())->value();
  pir::Value step_res = std::static_pointer_cast<LazyTensor>(step.impl())->value();
  auto op_res = paddle::dialect::arange(start_res, end_res, step_res, dtype, place);
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
Tensor assign_out_<LazyTensor>(const Tensor& x, const Tensor& output) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value output_res = std::static_pointer_cast<LazyTensor>(output.impl())->value();
  auto op_res = paddle::dialect::assign_out_(x_res, output_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor assign_value_<LazyTensor>(const Tensor& output, const std::vector<int>& shape, DataType dtype, const std::vector<Scalar>& values, Place place) {
  pir::Value output_res = std::static_pointer_cast<LazyTensor>(output.impl())->value();
  auto op_res = paddle::dialect::assign_value_(output_res, shape, dtype, values, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm<LazyTensor>(const Tensor& x, const Tensor& mean, const Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_layout, bool use_global_stats, bool trainable_statistics) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value mean_res = std::static_pointer_cast<LazyTensor>(mean.impl())->value();
  pir::Value variance_res = std::static_pointer_cast<LazyTensor>(variance.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  pir::Value bias_res = std::static_pointer_cast<LazyTensor>(bias.impl())->value();
  auto op_res = paddle::dialect::batch_norm(x_res, mean_res, variance_res, scale_res, bias_res, is_test, momentum, epsilon, data_layout, use_global_stats, trainable_statistics);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor mean_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor variance_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor saved_mean(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor saved_variance(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  Tensor reserve_space(std::make_shared<LazyTensor>(op_res_5));
  return std::make_tuple(out, mean_out, variance_out, saved_mean, saved_variance, reserve_space); 
}

template <>
Tensor c_allgather<LazyTensor>(const Tensor& x, int ring_id, int nranks, bool use_calc_stream) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_allgather(x_res, ring_id, nranks, use_calc_stream);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_allreduce_max<LazyTensor>(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_allreduce_max(x_res, ring_id, use_calc_stream, use_model_parallel);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_allreduce_sum<LazyTensor>(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_allreduce_sum(x_res, ring_id, use_calc_stream, use_model_parallel);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_broadcast<LazyTensor>(const Tensor& x, int ring_id, int root, bool use_calc_stream) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_broadcast(x_res, ring_id, root, use_calc_stream);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_concat<LazyTensor>(const Tensor& x, int rank, int nranks, int ring_id, bool use_calc_stream, bool use_model_parallel) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_concat(x_res, rank, nranks, ring_id, use_calc_stream, use_model_parallel);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_embedding<LazyTensor>(const Tensor& weight, const Tensor& x, int64_t start_index) {
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_embedding(weight_res, x_res, start_index);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_identity<LazyTensor>(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_identity(x_res, ring_id, use_calc_stream, use_model_parallel);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_reduce_sum<LazyTensor>(const Tensor& x, int ring_id, int root_id, bool use_calc_stream) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_reduce_sum(x_res, ring_id, root_id, use_calc_stream);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_sync_calc_stream<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_sync_calc_stream(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_sync_comm_stream<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_sync_comm_stream(x_res);
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
Tensor channel_shuffle<LazyTensor>(const Tensor& x, int groups, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::channel_shuffle(x_res, groups, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor conv2d_transpose<LazyTensor>(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  auto op_res = paddle::dialect::conv2d_transpose(x_res, filter_res, strides, paddings, output_padding, output_size.GetData(), padding_algorithm, groups, dilations, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor conv2d_transpose<LazyTensor>(const Tensor& x, const Tensor& filter, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::OpResult output_size_res = std::static_pointer_cast<LazyTensor>(output_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::conv2d_transpose(x_res, filter_res, output_size_res, strides, paddings, output_padding, padding_algorithm, groups, dilations, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor decode_jpeg<LazyTensor>(const Tensor& x, const std::string& mode, Place place) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::decode_jpeg(x_res, mode, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor deformable_conv<LazyTensor>(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value offset_res = std::static_pointer_cast<LazyTensor>(offset.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  paddle::optional<pir::Value> mask_res;
  if(mask) {
    pir::Value mask_res_inner;
    mask_res_inner = std::static_pointer_cast<LazyTensor>(mask.get().impl())->value();
    mask_res = paddle::make_optional<pir::Value>(mask_res_inner);
  }
  auto op_res = paddle::dialect::deformable_conv(x_res, offset_res, filter_res, mask_res, strides, paddings, dilations, deformable_groups, groups, im2col_step);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor depthwise_conv2d_transpose<LazyTensor>(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  auto op_res = paddle::dialect::depthwise_conv2d_transpose(x_res, filter_res, strides, paddings, output_padding, output_size.GetData(), padding_algorithm, groups, dilations, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor depthwise_conv2d_transpose<LazyTensor>(const Tensor& x, const Tensor& filter, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::OpResult output_size_res = std::static_pointer_cast<LazyTensor>(output_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::depthwise_conv2d_transpose(x_res, filter_res, output_size_res, strides, paddings, output_padding, padding_algorithm, groups, dilations, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor disable_check_model_nan_inf<LazyTensor>(const Tensor& x, int flag) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::disable_check_model_nan_inf(x_res, flag);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, Tensor> distribute_fpn_proposals<LazyTensor>(const Tensor& fpn_rois, const paddle::optional<Tensor>& rois_num, int min_level, int max_level, int refer_level, int refer_scale, bool pixel_offset) {
  pir::Value fpn_rois_res = std::static_pointer_cast<LazyTensor>(fpn_rois.impl())->value();
  paddle::optional<pir::Value> rois_num_res;
  if(rois_num) {
    pir::Value rois_num_res_inner;
    rois_num_res_inner = std::static_pointer_cast<LazyTensor>(rois_num.get().impl())->value();
    rois_num_res = paddle::make_optional<pir::Value>(rois_num_res_inner);
  }
  auto op_res = paddle::dialect::distribute_fpn_proposals(fpn_rois_res, rois_num_res, min_level, max_level, refer_level, refer_scale, pixel_offset);
  auto op_res_0 = std::get<0>(op_res);
  std::vector<Tensor> multi_fpn_rois(op_res_0.size());
  std::transform(op_res_0.begin(), op_res_0.end(), multi_fpn_rois.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_1 = std::get<1>(op_res);
  std::vector<Tensor> multi_level_rois_num(op_res_1.size());
  std::transform(op_res_1.begin(), op_res_1.end(), multi_level_rois_num.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_2 = std::get<2>(op_res);
  Tensor restore_index(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(multi_fpn_rois, multi_level_rois_num, restore_index); 
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
std::tuple<Tensor, Tensor> dropout<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& seed_tensor, const Scalar& p, bool is_test, const std::string& mode, int seed, bool fix_seed) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> seed_tensor_res;
  if(seed_tensor) {
    pir::Value seed_tensor_res_inner;
    seed_tensor_res_inner = std::static_pointer_cast<LazyTensor>(seed_tensor.get().impl())->value();
    seed_tensor_res = paddle::make_optional<pir::Value>(seed_tensor_res_inner);
  }
  auto op_res = paddle::dialect::dropout(x_res, seed_tensor_res, p.to<float>(), is_test, mode, seed, fix_seed);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor mask(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, mask); 
}

template <>
std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> einsum<LazyTensor>(const std::vector<Tensor>& x, const std::string& equation) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::einsum(x_res, equation);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  std::vector<Tensor> inner_cache(op_res_1.size());
  std::transform(op_res_1.begin(), op_res_1.end(), inner_cache.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_2 = std::get<2>(op_res);
  std::vector<Tensor> xshape(op_res_2.size());
  std::transform(op_res_2.begin(), op_res_2.end(), xshape.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return std::make_tuple(out, inner_cache, xshape); 
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
Tensor embedding<LazyTensor>(const Tensor& x, const Tensor& weight, int64_t padding_idx, bool sparse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  auto op_res = paddle::dialect::embedding(x_res, weight_res, padding_idx, sparse);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor embedding_grad_dense<LazyTensor>(const Tensor& x, const Tensor& weight, const Tensor& out_grad, int64_t padding_idx, bool sparse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::embedding_grad_dense(x_res, weight_res, out_grad_res, padding_idx, sparse);
  Tensor weight_grad(std::make_shared<LazyTensor>(op_res));
  return weight_grad; 
}

template <>
Tensor empty<LazyTensor>(const IntArray& shape, DataType dtype, Place place) {
  auto op_res = paddle::dialect::empty(shape.GetData(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor empty<LazyTensor>(const Tensor& shape_, DataType dtype, Place place) {
  pir::OpResult shape_res = std::static_pointer_cast<LazyTensor>(shape_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::empty(shape_res, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor empty_like<LazyTensor>(const Tensor& x, DataType dtype, Place place) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::empty_like(x_res, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor enable_check_model_nan_inf<LazyTensor>(const Tensor& x, int flag) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::enable_check_model_nan_inf(x_res, flag);
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
Tensor exponential_<LazyTensor>(const Tensor& x, float lam) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::exponential_(x_res, lam);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor eye<LazyTensor>(const Scalar& num_rows, const Scalar& num_columns, DataType dtype, Place place) {
  auto op_res = paddle::dialect::eye(num_rows.to<float>(), num_columns.to<float>(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor eye<LazyTensor>(const Tensor& num_rows_, const Tensor& num_columns_, DataType dtype, Place place) {
  pir::OpResult num_rows_res = std::static_pointer_cast<LazyTensor>(num_rows_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult num_columns_res = std::static_pointer_cast<LazyTensor>(num_columns_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::eye(num_rows_res, num_columns_res, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor floor_divide<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::floor_divide(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor frobenius_norm<LazyTensor>(const Tensor& x, const std::vector<int64_t>& axis, bool keep_dim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::frobenius_norm(x_res, axis, keep_dim, reduce_all);
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
Tensor full_<LazyTensor>(const Tensor& output, const IntArray& shape, const Scalar& value, DataType dtype, Place place) {
  pir::Value output_res = std::static_pointer_cast<LazyTensor>(output.impl())->value();
  auto op_res = paddle::dialect::full_(output_res, shape.GetData(), value.to<float>(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor full_batch_size_like<LazyTensor>(const Tensor& input, const std::vector<int>& shape, DataType dtype, const Scalar& value, int input_dim_idx, int output_dim_idx, Place place) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::full_batch_size_like(input_res, shape, dtype, value.to<float>(), input_dim_idx, output_dim_idx, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor full_like<LazyTensor>(const Tensor& x, const Scalar& value, DataType dtype, Place place) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::full_like(x_res, value.to<float>(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor full_like<LazyTensor>(const Tensor& x, const Tensor& value_, DataType dtype, Place place) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult value_res = std::static_pointer_cast<LazyTensor>(value_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::full_like(x_res, value_res, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor full_with_tensor<LazyTensor>(const Tensor& shape, const Tensor& value, DataType dtype) {
  pir::Value shape_res = std::static_pointer_cast<LazyTensor>(shape.impl())->value();
  pir::Value value_res = std::static_pointer_cast<LazyTensor>(value.impl())->value();
  auto op_res = paddle::dialect::full_with_tensor(shape_res, value_res, dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>, const paddle::optional<std::vector<Tensor>>> fused_adam_<LazyTensor>(const std::vector<Tensor>& params, const std::vector<Tensor>& grads, const Tensor& learning_rate, const std::vector<Tensor>& moments1, const std::vector<Tensor>& moments2, const std::vector<Tensor>& beta1_pows, const std::vector<Tensor>& beta2_pows, const paddle::optional<std::vector<Tensor>>& master_params, const paddle::optional<Tensor>& skip_update, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, int chunk_size, float weight_decay, bool use_adamw, bool multi_precision, bool use_global_beta_pow) {
  std::vector<pir::Value> params_res(params.size());
  std::transform(params.begin(), params.end(), params_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> grads_res(grads.size());
  std::transform(grads.begin(), grads.end(), grads_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value learning_rate_res = std::static_pointer_cast<LazyTensor>(learning_rate.impl())->value();
  std::vector<pir::Value> moments1_res(moments1.size());
  std::transform(moments1.begin(), moments1.end(), moments1_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> moments2_res(moments2.size());
  std::transform(moments2.begin(), moments2.end(), moments2_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> beta1_pows_res(beta1_pows.size());
  std::transform(beta1_pows.begin(), beta1_pows.end(), beta1_pows_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> beta2_pows_res(beta2_pows.size());
  std::transform(beta2_pows.begin(), beta2_pows.end(), beta2_pows_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  paddle::optional<std::vector<pir::Value>> master_params_res;
  if(master_params) {
    std::vector<pir::Value> master_params_res_inner(master_params.get().size());
    std::transform(master_params.get().begin(), master_params.get().end(), master_params_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    master_params_res = paddle::make_optional<std::vector<pir::Value>>(master_params_res_inner);
  }
  paddle::optional<pir::Value> skip_update_res;
  if(skip_update) {
    pir::Value skip_update_res_inner;
    skip_update_res_inner = std::static_pointer_cast<LazyTensor>(skip_update.get().impl())->value();
    skip_update_res = paddle::make_optional<pir::Value>(skip_update_res_inner);
  }
  auto op_res = paddle::dialect::fused_adam_(params_res, grads_res, learning_rate_res, moments1_res, moments2_res, beta1_pows_res, beta2_pows_res, master_params_res, skip_update_res, beta1.to<float>(), beta2.to<float>(), epsilon.to<float>(), chunk_size, weight_decay, use_adamw, multi_precision, use_global_beta_pow);
  auto op_res_0 = std::get<0>(op_res);
  std::vector<Tensor> params_out(op_res_0.size());
  std::transform(op_res_0.begin(), op_res_0.end(), params_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_1 = std::get<1>(op_res);
  std::vector<Tensor> moments1_out(op_res_1.size());
  std::transform(op_res_1.begin(), op_res_1.end(), moments1_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_2 = std::get<2>(op_res);
  std::vector<Tensor> moments2_out(op_res_2.size());
  std::transform(op_res_2.begin(), op_res_2.end(), moments2_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_3 = std::get<3>(op_res);
  std::vector<Tensor> beta1_pows_out(op_res_3.size());
  std::transform(op_res_3.begin(), op_res_3.end(), beta1_pows_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_4 = std::get<4>(op_res);
  std::vector<Tensor> beta2_pows_out(op_res_4.size());
  std::transform(op_res_4.begin(), op_res_4.end(), beta2_pows_out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_5 = std::get<5>(op_res);
  paddle::optional<std::vector<Tensor>> master_params_out;
  if(op_res_5){
    std::vector<Tensor> master_params_out_inner(op_res_5.get().size());
    std::transform(op_res_5.get().begin(), op_res_5.get().end(), master_params_out_inner.begin(), [](const pir::OpResult& res) {
      return Tensor(std::make_shared<LazyTensor>(res));
    });
    master_params_out = paddle::make_optional<std::vector<Tensor>>(master_params_out_inner);
  }
  return std::make_tuple(params_out, moments1_out, moments2_out, beta1_pows_out, beta2_pows_out, master_params_out); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> fused_batch_norm_act<LazyTensor>(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& variance, float momentum, float epsilon, const std::string& act_type) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  pir::Value bias_res = std::static_pointer_cast<LazyTensor>(bias.impl())->value();
  pir::Value mean_res = std::static_pointer_cast<LazyTensor>(mean.impl())->value();
  pir::Value variance_res = std::static_pointer_cast<LazyTensor>(variance.impl())->value();
  auto op_res = paddle::dialect::fused_batch_norm_act(x_res, scale_res, bias_res, mean_res, variance_res, momentum, epsilon, act_type);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor mean_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor variance_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor saved_mean(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor saved_variance(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  Tensor reserve_space(std::make_shared<LazyTensor>(op_res_5));
  return std::make_tuple(out, mean_out, variance_out, saved_mean, saved_variance, reserve_space); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> fused_bn_add_activation<LazyTensor>(const Tensor& x, const Tensor& z, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& variance, float momentum, float epsilon, const std::string& act_type) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value z_res = std::static_pointer_cast<LazyTensor>(z.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  pir::Value bias_res = std::static_pointer_cast<LazyTensor>(bias.impl())->value();
  pir::Value mean_res = std::static_pointer_cast<LazyTensor>(mean.impl())->value();
  pir::Value variance_res = std::static_pointer_cast<LazyTensor>(variance.impl())->value();
  auto op_res = paddle::dialect::fused_bn_add_activation(x_res, z_res, scale_res, bias_res, mean_res, variance_res, momentum, epsilon, act_type);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor mean_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor variance_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor saved_mean(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor saved_variance(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  Tensor reserve_space(std::make_shared<LazyTensor>(op_res_5));
  return std::make_tuple(out, mean_out, variance_out, saved_mean, saved_variance, reserve_space); 
}

template <>
Tensor fused_softmax_mask_upper_triangle<LazyTensor>(const Tensor& X) {
  pir::Value X_res = std::static_pointer_cast<LazyTensor>(X.impl())->value();
  auto op_res = paddle::dialect::fused_softmax_mask_upper_triangle(X_res);
  Tensor Out(std::make_shared<LazyTensor>(op_res));
  return Out; 
}

template <>
Tensor gaussian<LazyTensor>(const IntArray& shape, float mean, float std, int seed, DataType dtype, Place place) {
  auto op_res = paddle::dialect::gaussian(shape.GetData(), mean, std, seed, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor gaussian<LazyTensor>(const Tensor& shape_, float mean, float std, int seed, DataType dtype, Place place) {
  pir::OpResult shape_res = std::static_pointer_cast<LazyTensor>(shape_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::gaussian(shape_res, mean, std, seed, dtype, place);
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
Tensor hardswish<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::hardswish(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> hsigmoid_loss<LazyTensor>(const Tensor& x, const Tensor& label, const Tensor& w, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, int num_classes, bool is_sparse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value w_res = std::static_pointer_cast<LazyTensor>(w.impl())->value();
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  paddle::optional<pir::Value> path_res;
  if(path) {
    pir::Value path_res_inner;
    path_res_inner = std::static_pointer_cast<LazyTensor>(path.get().impl())->value();
    path_res = paddle::make_optional<pir::Value>(path_res_inner);
  }
  paddle::optional<pir::Value> code_res;
  if(code) {
    pir::Value code_res_inner;
    code_res_inner = std::static_pointer_cast<LazyTensor>(code.get().impl())->value();
    code_res = paddle::make_optional<pir::Value>(code_res_inner);
  }
  auto op_res = paddle::dialect::hsigmoid_loss(x_res, label_res, w_res, bias_res, path_res, code_res, num_classes, is_sparse);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor pre_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor w_out(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out, pre_out, w_out); 
}

template <>
Tensor increment<LazyTensor>(const Tensor& x, float value) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::increment(x_res, value);
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
Tensor linspace<LazyTensor>(const Tensor& start, const Tensor& stop, const Tensor& number, DataType dtype, Place place) {
  pir::Value start_res = std::static_pointer_cast<LazyTensor>(start.impl())->value();
  pir::Value stop_res = std::static_pointer_cast<LazyTensor>(stop.impl())->value();
  pir::Value number_res = std::static_pointer_cast<LazyTensor>(number.impl())->value();
  auto op_res = paddle::dialect::linspace(start_res, stop_res, number_res, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logspace<LazyTensor>(const Tensor& start, const Tensor& stop, const Tensor& num, const Tensor& base, DataType dtype, Place place) {
  pir::Value start_res = std::static_pointer_cast<LazyTensor>(start.impl())->value();
  pir::Value stop_res = std::static_pointer_cast<LazyTensor>(stop.impl())->value();
  pir::Value num_res = std::static_pointer_cast<LazyTensor>(num.impl())->value();
  pir::Value base_res = std::static_pointer_cast<LazyTensor>(base.impl())->value();
  auto op_res = paddle::dialect::logspace(start_res, stop_res, num_res, base_res, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logsumexp<LazyTensor>(const Tensor& x, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::logsumexp(x_res, axis, keepdim, reduce_all);
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
Tensor matrix_rank<LazyTensor>(const Tensor& x, float tol, bool use_default_tol, bool hermitian) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::matrix_rank(x_res, tol, use_default_tol, hermitian);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor matrix_rank_tol<LazyTensor>(const Tensor& x, const Tensor& atol_tensor, bool use_default_tol, bool hermitian) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value atol_tensor_res = std::static_pointer_cast<LazyTensor>(atol_tensor.impl())->value();
  auto op_res = paddle::dialect::matrix_rank_tol(x_res, atol_tensor_res, use_default_tol, hermitian);
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
Tensor memcpy_d2h<LazyTensor>(const Tensor& x, int dst_place_type) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::memcpy_d2h(x_res, dst_place_type);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor memcpy_h2d<LazyTensor>(const Tensor& x, int dst_place_type) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::memcpy_h2d(x_res, dst_place_type);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor min<LazyTensor>(const Tensor& x, const IntArray& axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::min(x_res, axis.GetData(), keepdim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor min<LazyTensor>(const Tensor& x, const Tensor& axis_, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::min(x_res, axis_res, keepdim);
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
Tensor mish<LazyTensor>(const Tensor& x, float lambda) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::mish(x_res, lambda);
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
std::tuple<Tensor, Tensor> norm<LazyTensor>(const Tensor& x, int axis, float epsilon, bool is_test) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::norm(x_res, axis, epsilon, is_test);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor norm(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out, norm); 
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
Tensor one_hot<LazyTensor>(const Tensor& x, const Scalar& num_classes) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::one_hot(x_res, num_classes.to<int>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor ones<LazyTensor>(const IntArray& shape, DataType dtype, Place place) {
  auto op_res = paddle::dialect::ones(shape.GetData(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor ones_like<LazyTensor>(const Tensor& x, DataType dtype, Place place) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::ones_like(x_res, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pad<LazyTensor>(const Tensor& x, const std::vector<int>& paddings, const Scalar& pad_value) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::pad(x_res, paddings, pad_value.to<float>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pool2d<LazyTensor>(const Tensor& x, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::pool2d(x_res, kernel_size.GetData(), strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pool2d<LazyTensor>(const Tensor& x, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult kernel_size_res = std::static_pointer_cast<LazyTensor>(kernel_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::pool2d(x_res, kernel_size_res, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pool3d<LazyTensor>(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::pool3d(x_res, kernel_size, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor prod<LazyTensor>(const Tensor& x, const IntArray& dims, bool keep_dim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::prod(x_res, dims.GetData(), keep_dim, reduce_all);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor prod<LazyTensor>(const Tensor& x, const Tensor& dims_, bool keep_dim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult dims_res = std::static_pointer_cast<LazyTensor>(dims_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::prod(x_res, dims_res, keep_dim, reduce_all);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor randint<LazyTensor>(int low, int high, const IntArray& shape, DataType dtype, Place place) {
  auto op_res = paddle::dialect::randint(low, high, shape.GetData(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor randint<LazyTensor>(const Tensor& shape_, int low, int high, DataType dtype, Place place) {
  pir::OpResult shape_res = std::static_pointer_cast<LazyTensor>(shape_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::randint(shape_res, low, high, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor randperm<LazyTensor>(int n, DataType dtype, Place place) {
  auto op_res = paddle::dialect::randperm(n, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor remainder<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::remainder(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor repeat_interleave<LazyTensor>(const Tensor& x, int repeats, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::repeat_interleave(x_res, repeats, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor repeat_interleave_with_tensor_index<LazyTensor>(const Tensor& x, const Tensor& repeats, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value repeats_res = std::static_pointer_cast<LazyTensor>(repeats.impl())->value();
  auto op_res = paddle::dialect::repeat_interleave_with_tensor_index(x_res, repeats_res, axis);
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
std::tuple<Tensor, Tensor, std::vector<Tensor>> rnn<LazyTensor>(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& dropout_state_in, float dropout_prob, bool is_bidirec, int input_size, int hidden_size, int num_layers, const std::string& mode, int seed, bool is_test) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  std::vector<pir::Value> pre_state_res(pre_state.size());
  std::transform(pre_state.begin(), pre_state.end(), pre_state_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> weight_list_res(weight_list.size());
  std::transform(weight_list.begin(), weight_list.end(), weight_list_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  paddle::optional<pir::Value> sequence_length_res;
  if(sequence_length) {
    pir::Value sequence_length_res_inner;
    sequence_length_res_inner = std::static_pointer_cast<LazyTensor>(sequence_length.get().impl())->value();
    sequence_length_res = paddle::make_optional<pir::Value>(sequence_length_res_inner);
  }
  pir::Value dropout_state_in_res = std::static_pointer_cast<LazyTensor>(dropout_state_in.impl())->value();
  auto op_res = paddle::dialect::rnn(x_res, pre_state_res, weight_list_res, sequence_length_res, dropout_state_in_res, dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode, seed, is_test);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor dropout_state_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  std::vector<Tensor> state(op_res_2.size());
  std::transform(op_res_2.begin(), op_res_2.end(), state.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return std::make_tuple(out, dropout_state_out, state); 
}

template <>
Tensor rrelu<LazyTensor>(const Tensor& x, float lower, float upper, bool is_test) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::rrelu(x_res, lower, upper, is_test);
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
Tensor softmax<LazyTensor>(const Tensor& x, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::softmax(x_res, axis);
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
std::vector<Tensor> split_with_num<LazyTensor>(const Tensor& x, int num, const Scalar& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::split_with_num(x_res, num, axis.to<int>());
  std::vector<Tensor> out(op_res.size());
  std::transform(op_res.begin(), op_res.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return out; 
}

template <>
std::vector<Tensor> split_with_num<LazyTensor>(const Tensor& x, const Tensor& axis_, int num) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::split_with_num(x_res, axis_res, num);
  std::vector<Tensor> out(op_res.size());
  std::transform(op_res.begin(), op_res.end(), out.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return out; 
}

template <>
Tensor strided_slice<LazyTensor>(const Tensor& x, const std::vector<int>& axes, const IntArray& starts, const IntArray& ends, const IntArray& strides) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::strided_slice(x_res, axes, starts.GetData(), ends.GetData(), strides.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor strided_slice<LazyTensor>(const Tensor& x, const Tensor& starts_, const Tensor& ends_, const Tensor& strides_, const std::vector<int>& axes) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult starts_res = std::static_pointer_cast<LazyTensor>(starts_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult ends_res = std::static_pointer_cast<LazyTensor>(ends_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult strides_res = std::static_pointer_cast<LazyTensor>(strides_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::strided_slice(x_res, starts_res, ends_res, strides_res, axes);
  Tensor out(std::make_shared<LazyTensor>(op_res));
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
Tensor swish<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::swish(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> sync_batch_norm_<LazyTensor>(const Tensor& x, const Tensor& mean, const Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_layout, bool use_global_stats, bool trainable_statistics) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value mean_res = std::static_pointer_cast<LazyTensor>(mean.impl())->value();
  pir::Value variance_res = std::static_pointer_cast<LazyTensor>(variance.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  pir::Value bias_res = std::static_pointer_cast<LazyTensor>(bias.impl())->value();
  auto op_res = paddle::dialect::sync_batch_norm_(x_res, mean_res, variance_res, scale_res, bias_res, is_test, momentum, epsilon, data_layout, use_global_stats, trainable_statistics);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor mean_out(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor variance_out(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor saved_mean(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor saved_variance(std::make_shared<LazyTensor>(op_res_4));
  auto op_res_5 = std::get<5>(op_res);
  Tensor reserve_space(std::make_shared<LazyTensor>(op_res_5));
  return std::make_tuple(out, mean_out, variance_out, saved_mean, saved_variance, reserve_space); 
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
Tensor trans_layout<LazyTensor>(const Tensor& x, const std::vector<int>& perm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::trans_layout(x_res, perm);
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
Tensor tril<LazyTensor>(const Tensor& x, int diagonal) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::tril(x_res, diagonal);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor tril_indices<LazyTensor>(int rows, int cols, int offset, DataType dtype, Place place) {
  auto op_res = paddle::dialect::tril_indices(rows, cols, offset, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor triu<LazyTensor>(const Tensor& x, int diagonal) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::triu(x_res, diagonal);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor triu_indices<LazyTensor>(int row, int col, int offset, DataType dtype, Place place) {
  auto op_res = paddle::dialect::triu_indices(row, col, offset, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor truncated_gaussian_random<LazyTensor>(const std::vector<int>& shape, float mean, float std, int seed, DataType dtype, Place place) {
  auto op_res = paddle::dialect::truncated_gaussian_random(shape, mean, std, seed, dtype, place);
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
std::tuple<Tensor, Tensor, Tensor, Tensor> unique<LazyTensor>(const Tensor& x, bool return_index, bool return_inverse, bool return_counts, const std::vector<int>& axis, DataType dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::unique(x_res, return_index, return_inverse, return_counts, axis, dtype);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor indices(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor inverse(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor counts(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(out, indices, inverse, counts); 
}

template <>
Tensor unpool<LazyTensor>(const Tensor& x, const Tensor& indices, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const IntArray& output_size, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  auto op_res = paddle::dialect::unpool(x_res, indices_res, ksize, strides, padding, output_size.GetData(), data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor unpool<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& output_size_, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::OpResult output_size_res = std::static_pointer_cast<LazyTensor>(output_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::unpool(x_res, indices_res, output_size_res, ksize, strides, padding, data_format);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor zeros<LazyTensor>(const IntArray& shape, DataType dtype, Place place) {
  auto op_res = paddle::dialect::zeros(shape.GetData(), dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor zeros_like<LazyTensor>(const Tensor& x, DataType dtype, Place place) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::zeros_like(x_res, dtype, place);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor abs_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_x_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::abs_double_grad(x_res, grad_x_grad_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor abs_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::abs_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor acos_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::acos_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor acosh_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::acosh_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> addmm_grad<LazyTensor>(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha, float beta) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::addmm_grad(input_res, x_res, y_res, out_grad_res, alpha, beta);
  auto op_res_0 = std::get<0>(op_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(input_grad, x_grad, y_grad); 
}

template <>
Tensor affine_grid_grad<LazyTensor>(const Tensor& input, const Tensor& output_grad, const IntArray& output_shape, bool align_corners) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value output_grad_res = std::static_pointer_cast<LazyTensor>(output_grad.impl())->value();
  auto op_res = paddle::dialect::affine_grid_grad(input_res, output_grad_res, output_shape.GetData(), align_corners);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor affine_grid_grad<LazyTensor>(const Tensor& input, const Tensor& output_grad, const Tensor& output_shape_, bool align_corners) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value output_grad_res = std::static_pointer_cast<LazyTensor>(output_grad.impl())->value();
  pir::OpResult output_shape_res = std::static_pointer_cast<LazyTensor>(output_shape_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::affine_grid_grad(input_res, output_grad_res, output_shape_res, align_corners);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor angle_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::angle_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor argsort_grad<LazyTensor>(const Tensor& indices, const Tensor& x, const Tensor& out_grad, int axis, bool descending) {
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::argsort_grad(indices_res, x_res, out_grad_res, axis, descending);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor as_strided_grad<LazyTensor>(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims, const std::vector<int64_t>& stride, int64_t offset) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::as_strided_grad(input_res, out_grad_res, dims, stride, offset);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor asin_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::asin_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor asinh_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::asinh_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> atan2_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::atan2_grad(x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor atan_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::atan_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor atanh_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::atanh_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor bce_loss_grad<LazyTensor>(const Tensor& input, const Tensor& label, const Tensor& out_grad) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::bce_loss_grad(input_res, label_res, out_grad_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor bicubic_interp_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  pir::Value output_grad_res = std::static_pointer_cast<LazyTensor>(output_grad.impl())->value();
  auto op_res = paddle::dialect::bicubic_interp_grad(x_res, out_size_res, size_tensor_res, scale_tensor_res, output_grad_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor> bilinear_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::bilinear_grad(x_res, y_res, weight_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor weight_grad(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(x_grad, y_grad, weight_grad, bias_grad); 
}

template <>
Tensor bilinear_interp_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  pir::Value output_grad_res = std::static_pointer_cast<LazyTensor>(output_grad.impl())->value();
  auto op_res = paddle::dialect::bilinear_interp_grad(x_res, out_size_res, size_tensor_res, scale_tensor_res, output_grad_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> bmm_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::bmm_grad(x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
std::vector<Tensor> broadcast_tensors_grad<LazyTensor>(const std::vector<Tensor>& input, const std::vector<Tensor>& out_grad) {
  std::vector<pir::Value> input_res(input.size());
  std::transform(input.begin(), input.end(), input_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> out_grad_res(out_grad.size());
  std::transform(out_grad.begin(), out_grad.end(), out_grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::broadcast_tensors_grad(input_res, out_grad_res);
  std::vector<Tensor> input_grad(op_res.size());
  std::transform(op_res.begin(), op_res.end(), input_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return input_grad; 
}

template <>
Tensor ceil_grad<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::ceil_grad(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> celu_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::celu_double_grad(x_res, grad_out_res, grad_x_grad_res, alpha);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor celu_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::celu_grad(x_res, out_grad_res, alpha);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor cholesky_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad, bool upper) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cholesky_grad(out_res, out_grad_res, upper);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> cholesky_solve_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cholesky_solve_grad(x_res, y_res, out_res, out_grad_res, upper);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor clip_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_x_grad, const Scalar& min, const Scalar& max) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::clip_double_grad(x_res, grad_x_grad_res, min.to<float>(), max.to<float>());
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor clip_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_x_grad, const Tensor& min_, const Tensor& max_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  pir::OpResult min_res = std::static_pointer_cast<LazyTensor>(min_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult max_res = std::static_pointer_cast<LazyTensor>(max_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::clip_double_grad(x_res, grad_x_grad_res, min_res, max_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor clip_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Scalar& min, const Scalar& max) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::clip_grad(x_res, out_grad_res, min.to<float>(), max.to<float>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor clip_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& min_, const Tensor& max_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult min_res = std::static_pointer_cast<LazyTensor>(min_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult max_res = std::static_pointer_cast<LazyTensor>(max_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::clip_grad(x_res, out_grad_res, min_res, max_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> complex_grad<LazyTensor>(const Tensor& real, const Tensor& imag, const Tensor& out_grad) {
  pir::Value real_res = std::static_pointer_cast<LazyTensor>(real.impl())->value();
  pir::Value imag_res = std::static_pointer_cast<LazyTensor>(imag.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::complex_grad(real_res, imag_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor real_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor imag_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(real_grad, imag_grad); 
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
std::tuple<Tensor, Tensor> conv2d_grad<LazyTensor>(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::conv2d_grad(input_res, filter_res, out_grad_res, strides, paddings, padding_algorithm, dilations, groups, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(input_grad, filter_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> conv2d_grad_grad<LazyTensor>(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_input_grad_res;
  if(grad_input_grad) {
    pir::Value grad_input_grad_res_inner;
    grad_input_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_input_grad.get().impl())->value();
    grad_input_grad_res = paddle::make_optional<pir::Value>(grad_input_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_filter_grad_res;
  if(grad_filter_grad) {
    pir::Value grad_filter_grad_res_inner;
    grad_filter_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_filter_grad.get().impl())->value();
    grad_filter_grad_res = paddle::make_optional<pir::Value>(grad_filter_grad_res_inner);
  }
  auto op_res = paddle::dialect::conv2d_grad_grad(input_res, filter_res, grad_out_res, grad_input_grad_res, grad_filter_grad_res, strides, paddings, padding_algorithm, dilations, groups, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(input_grad, filter_grad, grad_out_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> conv3d_double_grad<LazyTensor>(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_input_grad_res;
  if(grad_input_grad) {
    pir::Value grad_input_grad_res_inner;
    grad_input_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_input_grad.get().impl())->value();
    grad_input_grad_res = paddle::make_optional<pir::Value>(grad_input_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_filter_grad_res;
  if(grad_filter_grad) {
    pir::Value grad_filter_grad_res_inner;
    grad_filter_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_filter_grad.get().impl())->value();
    grad_filter_grad_res = paddle::make_optional<pir::Value>(grad_filter_grad_res_inner);
  }
  auto op_res = paddle::dialect::conv3d_double_grad(input_res, filter_res, grad_out_res, grad_input_grad_res, grad_filter_grad_res, strides, paddings, padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(input_grad, filter_grad, grad_out_grad); 
}

template <>
std::tuple<Tensor, Tensor> conv3d_grad<LazyTensor>(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::conv3d_grad(input_res, filter_res, out_grad_res, strides, paddings, padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(input_grad, filter_grad); 
}

template <>
std::tuple<Tensor, Tensor> conv3d_transpose_grad<LazyTensor>(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::conv3d_transpose_grad(x_res, filter_res, out_grad_res, strides, paddings, output_padding, output_size, padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, filter_grad); 
}

template <>
std::tuple<Tensor, Tensor> cos_double_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> grad_out_res;
  if(grad_out) {
    pir::Value grad_out_res_inner;
    grad_out_res_inner = std::static_pointer_cast<LazyTensor>(grad_out.get().impl())->value();
    grad_out_res = paddle::make_optional<pir::Value>(grad_out_res_inner);
  }
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::cos_double_grad(x_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor cos_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cos_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> cos_triple_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> grad_out_forward_res;
  if(grad_out_forward) {
    pir::Value grad_out_forward_res_inner;
    grad_out_forward_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_forward.get().impl())->value();
    grad_out_forward_res = paddle::make_optional<pir::Value>(grad_out_forward_res_inner);
  }
  paddle::optional<pir::Value> grad_x_grad_forward_res;
  if(grad_x_grad_forward) {
    pir::Value grad_x_grad_forward_res_inner;
    grad_x_grad_forward_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad_forward.get().impl())->value();
    grad_x_grad_forward_res = paddle::make_optional<pir::Value>(grad_x_grad_forward_res_inner);
  }
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  paddle::optional<pir::Value> grad_out_grad_grad_res;
  if(grad_out_grad_grad) {
    pir::Value grad_out_grad_grad_res_inner;
    grad_out_grad_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_grad_grad.get().impl())->value();
    grad_out_grad_grad_res = paddle::make_optional<pir::Value>(grad_out_grad_grad_res_inner);
  }
  auto op_res = paddle::dialect::cos_triple_grad(x_res, grad_out_forward_res, grad_x_grad_forward_res, grad_x_grad_res, grad_out_grad_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_forward_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_x_grad_forward_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, grad_out_forward_grad, grad_x_grad_forward_grad); 
}

template <>
Tensor cosh_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cosh_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor crop_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const IntArray& offsets) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::crop_grad(x_res, out_grad_res, offsets.GetData());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor crop_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& offsets_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult offsets_res = std::static_pointer_cast<LazyTensor>(offsets_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::crop_grad(x_res, out_grad_res, offsets_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor cross_entropy_with_softmax_grad<LazyTensor>(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis) {
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value softmax_res = std::static_pointer_cast<LazyTensor>(softmax.impl())->value();
  pir::Value loss_grad_res = std::static_pointer_cast<LazyTensor>(loss_grad.impl())->value();
  auto op_res = paddle::dialect::cross_entropy_with_softmax_grad(label_res, softmax_res, loss_grad_res, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
std::tuple<Tensor, Tensor> cross_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cross_grad(x_res, y_res, out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor cummax_grad<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, int dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cummax_grad(x_res, indices_res, out_grad_res, axis, dtype);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor cummin_grad<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, int dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cummin_grad(x_res, indices_res, out_grad_res, axis, dtype);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor cumprod_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, int dim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cumprod_grad(x_res, out_res, out_grad_res, dim);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor cumsum_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Scalar& axis, bool flatten, bool exclusive, bool reverse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cumsum_grad(x_res, out_grad_res, axis.to<int>(), flatten, exclusive, reverse);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor cumsum_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool flatten, bool exclusive, bool reverse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::cumsum_grad(x_res, out_grad_res, axis_res, flatten, exclusive, reverse);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> depthwise_conv2d_double_grad<LazyTensor>(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_input_grad_res;
  if(grad_input_grad) {
    pir::Value grad_input_grad_res_inner;
    grad_input_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_input_grad.get().impl())->value();
    grad_input_grad_res = paddle::make_optional<pir::Value>(grad_input_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_filter_grad_res;
  if(grad_filter_grad) {
    pir::Value grad_filter_grad_res_inner;
    grad_filter_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_filter_grad.get().impl())->value();
    grad_filter_grad_res = paddle::make_optional<pir::Value>(grad_filter_grad_res_inner);
  }
  auto op_res = paddle::dialect::depthwise_conv2d_double_grad(input_res, filter_res, grad_out_res, grad_input_grad_res, grad_filter_grad_res, strides, paddings, padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(input_grad, filter_grad, grad_out_grad); 
}

template <>
std::tuple<Tensor, Tensor> depthwise_conv2d_grad<LazyTensor>(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::depthwise_conv2d_grad(input_res, filter_res, out_grad_res, strides, paddings, padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(input_grad, filter_grad); 
}

template <>
Tensor det_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::det_grad(x_res, out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor diag_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, int offset) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::diag_grad(x_res, out_grad_res, offset);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor diagonal_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, int offset, int axis1, int axis2) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::diagonal_grad(x_res, out_grad_res, offset, axis1, axis2);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor digamma_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::digamma_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> dist_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, float p) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::dist_grad(x_res, y_res, out_res, out_grad_res, p);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
std::tuple<Tensor, Tensor> dot_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::dot_grad(x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor eig_grad<LazyTensor>(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad) {
  pir::Value out_w_res = std::static_pointer_cast<LazyTensor>(out_w.impl())->value();
  pir::Value out_v_res = std::static_pointer_cast<LazyTensor>(out_v.impl())->value();
  pir::Value out_w_grad_res = std::static_pointer_cast<LazyTensor>(out_w_grad.impl())->value();
  pir::Value out_v_grad_res = std::static_pointer_cast<LazyTensor>(out_v_grad.impl())->value();
  auto op_res = paddle::dialect::eig_grad(out_w_res, out_v_res, out_w_grad_res, out_v_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor eigh_grad<LazyTensor>(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad) {
  pir::Value out_w_res = std::static_pointer_cast<LazyTensor>(out_w.impl())->value();
  pir::Value out_v_res = std::static_pointer_cast<LazyTensor>(out_v.impl())->value();
  pir::Value out_w_grad_res = std::static_pointer_cast<LazyTensor>(out_w_grad.impl())->value();
  pir::Value out_v_grad_res = std::static_pointer_cast<LazyTensor>(out_v_grad.impl())->value();
  auto op_res = paddle::dialect::eigh_grad(out_w_res, out_v_res, out_w_grad_res, out_v_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor eigvalsh_grad<LazyTensor>(const Tensor& eigenvectors, const Tensor& eigenvalues_grad, const std::string& uplo, bool is_test) {
  pir::Value eigenvectors_res = std::static_pointer_cast<LazyTensor>(eigenvectors.impl())->value();
  pir::Value eigenvalues_grad_res = std::static_pointer_cast<LazyTensor>(eigenvalues_grad.impl())->value();
  auto op_res = paddle::dialect::eigvalsh_grad(eigenvectors_res, eigenvalues_grad_res, uplo, is_test);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> elu_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::elu_double_grad(x_res, grad_out_res, grad_x_grad_res, alpha);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor elu_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::elu_grad(x_res, out_res, out_grad_res, alpha);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
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
Tensor erfinv_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::erfinv_grad(out_res, out_grad_res);
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
Tensor expand_as_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const std::vector<int>& target_shape) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::expand_as_grad(x_res, out_grad_res, target_shape);
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
Tensor expm1_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::expm1_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fft_c2c_grad<LazyTensor>(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fft_c2c_grad(out_grad_res, axes, normalization, forward);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fft_c2r_grad<LazyTensor>(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, int64_t last_dim_size) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fft_c2r_grad(out_grad_res, axes, normalization, forward, last_dim_size);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fft_r2c_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, bool onesided) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fft_r2c_grad(x_res, out_grad_res, axes, normalization, forward, onesided);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fill_diagonal_grad<LazyTensor>(const Tensor& out_grad, float value, int offset, bool wrap) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fill_diagonal_grad(out_grad_res, value, offset, wrap);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fill_diagonal_tensor_grad<LazyTensor>(const Tensor& out_grad, int64_t offset, int dim1, int dim2) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fill_diagonal_tensor_grad(out_grad_res, offset, dim1, dim2);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fill_grad<LazyTensor>(const Tensor& out_grad, const Scalar& value) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fill_grad(out_grad_res, value.to<float>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fill_grad<LazyTensor>(const Tensor& out_grad, const Tensor& value_) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult value_res = std::static_pointer_cast<LazyTensor>(value_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::fill_grad(out_grad_res, value_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> flash_attn_grad<LazyTensor>(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, float dropout, bool causal) {
  pir::Value q_res = std::static_pointer_cast<LazyTensor>(q.impl())->value();
  pir::Value k_res = std::static_pointer_cast<LazyTensor>(k.impl())->value();
  pir::Value v_res = std::static_pointer_cast<LazyTensor>(v.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value softmax_lse_res = std::static_pointer_cast<LazyTensor>(softmax_lse.impl())->value();
  pir::Value seed_offset_res = std::static_pointer_cast<LazyTensor>(seed_offset.impl())->value();
  paddle::optional<pir::Value> attn_mask_res;
  if(attn_mask) {
    pir::Value attn_mask_res_inner;
    attn_mask_res_inner = std::static_pointer_cast<LazyTensor>(attn_mask.get().impl())->value();
    attn_mask_res = paddle::make_optional<pir::Value>(attn_mask_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::flash_attn_grad(q_res, k_res, v_res, out_res, softmax_lse_res, seed_offset_res, attn_mask_res, out_grad_res, dropout, causal);
  auto op_res_0 = std::get<0>(op_res);
  Tensor q_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor k_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor v_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(q_grad, k_grad, v_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> flash_attn_unpadded_grad<LazyTensor>(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout, bool causal) {
  pir::Value q_res = std::static_pointer_cast<LazyTensor>(q.impl())->value();
  pir::Value k_res = std::static_pointer_cast<LazyTensor>(k.impl())->value();
  pir::Value v_res = std::static_pointer_cast<LazyTensor>(v.impl())->value();
  pir::Value cu_seqlens_q_res = std::static_pointer_cast<LazyTensor>(cu_seqlens_q.impl())->value();
  pir::Value cu_seqlens_k_res = std::static_pointer_cast<LazyTensor>(cu_seqlens_k.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value softmax_lse_res = std::static_pointer_cast<LazyTensor>(softmax_lse.impl())->value();
  pir::Value seed_offset_res = std::static_pointer_cast<LazyTensor>(seed_offset.impl())->value();
  paddle::optional<pir::Value> attn_mask_res;
  if(attn_mask) {
    pir::Value attn_mask_res_inner;
    attn_mask_res_inner = std::static_pointer_cast<LazyTensor>(attn_mask.get().impl())->value();
    attn_mask_res = paddle::make_optional<pir::Value>(attn_mask_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::flash_attn_unpadded_grad(q_res, k_res, v_res, cu_seqlens_q_res, cu_seqlens_k_res, out_res, softmax_lse_res, seed_offset_res, attn_mask_res, out_grad_res, max_seqlen_q, max_seqlen_k, scale, dropout, causal);
  auto op_res_0 = std::get<0>(op_res);
  Tensor q_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor k_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor v_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(q_grad, k_grad, v_grad); 
}

template <>
Tensor flatten_grad<LazyTensor>(const Tensor& xshape, const Tensor& out_grad) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::flatten_grad(xshape_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor floor_grad<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::floor_grad(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> fmax_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fmax_grad(x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
std::tuple<Tensor, Tensor> fmin_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fmin_grad(x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor fold_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const std::vector<int>& output_sizes, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fold_grad(x_res, out_grad_res, output_sizes, kernel_sizes, strides, paddings, dilations);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor frame_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, int frame_length, int hop_length, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::frame_grad(x_res, out_grad_res, frame_length, hop_length, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor gather_grad<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& out_grad, const Scalar& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::gather_grad(x_res, index_res, out_grad_res, axis.to<int>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor gather_grad<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& out_grad, const Tensor& axis_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::gather_grad(x_res, index_res, out_grad_res, axis_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor gather_nd_grad<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::gather_nd_grad(x_res, index_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor gaussian_inplace_grad<LazyTensor>(const Tensor& out_grad, float mean, float std, int seed) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::gaussian_inplace_grad(out_grad_res, mean, std, seed);
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
std::tuple<Tensor, Tensor> grid_sample_grad<LazyTensor>(const Tensor& x, const Tensor& grid, const Tensor& out_grad, const std::string& mode, const std::string& padding_mode, bool align_corners) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grid_res = std::static_pointer_cast<LazyTensor>(grid.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::grid_sample_grad(x_res, grid_res, out_grad_res, mode, padding_mode, align_corners);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grid_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grid_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> group_norm_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& y, const Tensor& mean, const Tensor& variance, const Tensor& y_grad, float epsilon, int groups, const std::string& data_layout) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> scale_res;
  if(scale) {
    pir::Value scale_res_inner;
    scale_res_inner = std::static_pointer_cast<LazyTensor>(scale.get().impl())->value();
    scale_res = paddle::make_optional<pir::Value>(scale_res_inner);
  }
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value mean_res = std::static_pointer_cast<LazyTensor>(mean.impl())->value();
  pir::Value variance_res = std::static_pointer_cast<LazyTensor>(variance.impl())->value();
  pir::Value y_grad_res = std::static_pointer_cast<LazyTensor>(y_grad.impl())->value();
  auto op_res = paddle::dialect::group_norm_grad(x_res, scale_res, bias_res, y_res, mean_res, variance_res, y_grad_res, epsilon, groups, data_layout);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, scale_grad, bias_grad); 
}

template <>
Tensor gumbel_softmax_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad, int axis) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::gumbel_softmax_grad(out_res, out_grad_res, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor hardshrink_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::hardshrink_grad(x_res, out_grad_res, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor hardsigmoid_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad, float slope, float offset) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::hardsigmoid_grad(out_res, out_grad_res, slope, offset);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor hardtanh_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float t_min, float t_max) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::hardtanh_grad(x_res, out_grad_res, t_min, t_max);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> heaviside_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::heaviside_grad(x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
std::tuple<Tensor, Tensor> huber_loss_grad<LazyTensor>(const Tensor& residual, const Tensor& out_grad, float delta) {
  pir::Value residual_res = std::static_pointer_cast<LazyTensor>(residual.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::huber_loss_grad(residual_res, out_grad_res, delta);
  auto op_res_0 = std::get<0>(op_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor label_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(input_grad, label_grad); 
}

template <>
Tensor i0_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::i0_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor i0e_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::i0e_grad(x_res, out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor i1_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::i1_grad(x_res, out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor i1e_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::i1e_grad(x_res, out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor imag_grad<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::imag_grad(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> index_add_grad<LazyTensor>(const Tensor& index, const Tensor& add_value, const Tensor& out_grad, int axis) {
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value add_value_res = std::static_pointer_cast<LazyTensor>(add_value.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::index_add_grad(index_res, add_value_res, out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor add_value_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, add_value_grad); 
}

template <>
std::tuple<Tensor, Tensor> index_put_grad<LazyTensor>(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, const Tensor& out_grad, bool accumulate) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  std::vector<pir::Value> indices_res(indices.size());
  std::transform(indices.begin(), indices.end(), indices_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value value_res = std::static_pointer_cast<LazyTensor>(value.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::index_put_grad(x_res, indices_res, value_res, out_grad_res, accumulate);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor value_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, value_grad); 
}

template <>
Tensor index_sample_grad<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::index_sample_grad(x_res, index_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor index_select_grad<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::index_select_grad(x_res, index_res, out_grad_res, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor index_select_strided_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, int64_t index, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::index_select_strided_grad(x_res, out_grad_res, index, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> instance_norm_double_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& fwd_scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float epsilon) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> fwd_scale_res;
  if(fwd_scale) {
    pir::Value fwd_scale_res_inner;
    fwd_scale_res_inner = std::static_pointer_cast<LazyTensor>(fwd_scale.get().impl())->value();
    fwd_scale_res = paddle::make_optional<pir::Value>(fwd_scale_res_inner);
  }
  pir::Value saved_mean_res = std::static_pointer_cast<LazyTensor>(saved_mean.impl())->value();
  pir::Value saved_variance_res = std::static_pointer_cast<LazyTensor>(saved_variance.impl())->value();
  pir::Value grad_y_res = std::static_pointer_cast<LazyTensor>(grad_y.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_scale_grad_res;
  if(grad_scale_grad) {
    pir::Value grad_scale_grad_res_inner;
    grad_scale_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_scale_grad.get().impl())->value();
    grad_scale_grad_res = paddle::make_optional<pir::Value>(grad_scale_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_bias_grad_res;
  if(grad_bias_grad) {
    pir::Value grad_bias_grad_res_inner;
    grad_bias_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_bias_grad.get().impl())->value();
    grad_bias_grad_res = paddle::make_optional<pir::Value>(grad_bias_grad_res_inner);
  }
  auto op_res = paddle::dialect::instance_norm_double_grad(x_res, fwd_scale_res, saved_mean_res, saved_variance_res, grad_y_res, grad_x_grad_res, grad_scale_grad_res, grad_bias_grad_res, epsilon);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor fwd_scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_y_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, fwd_scale_grad, grad_y_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> instance_norm_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& y_grad, float epsilon) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> scale_res;
  if(scale) {
    pir::Value scale_res_inner;
    scale_res_inner = std::static_pointer_cast<LazyTensor>(scale.get().impl())->value();
    scale_res = paddle::make_optional<pir::Value>(scale_res_inner);
  }
  pir::Value saved_mean_res = std::static_pointer_cast<LazyTensor>(saved_mean.impl())->value();
  pir::Value saved_variance_res = std::static_pointer_cast<LazyTensor>(saved_variance.impl())->value();
  pir::Value y_grad_res = std::static_pointer_cast<LazyTensor>(y_grad.impl())->value();
  auto op_res = paddle::dialect::instance_norm_grad(x_res, scale_res, saved_mean_res, saved_variance_res, y_grad_res, epsilon);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, scale_grad, bias_grad); 
}

template <>
Tensor inverse_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::inverse_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor kldiv_loss_grad<LazyTensor>(const Tensor& x, const Tensor& label, const Tensor& out_grad, const std::string& reduction) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::kldiv_loss_grad(x_res, label_res, out_grad_res, reduction);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> kron_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::kron_grad(x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor kthvalue_grad<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int k, int axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::kthvalue_grad(x_res, indices_res, out_grad_res, k, axis, keepdim);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor label_smooth_grad<LazyTensor>(const Tensor& out_grad, float epsilon) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::label_smooth_grad(out_grad_res, epsilon);
  Tensor label_grad(std::make_shared<LazyTensor>(op_res));
  return label_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> layer_norm_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& mean, const Tensor& variance, const Tensor& out_grad, float epsilon, int begin_norm_axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> scale_res;
  if(scale) {
    pir::Value scale_res_inner;
    scale_res_inner = std::static_pointer_cast<LazyTensor>(scale.get().impl())->value();
    scale_res = paddle::make_optional<pir::Value>(scale_res_inner);
  }
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
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
Tensor leaky_relu_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_x_grad, float negative_slope) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::leaky_relu_double_grad(x_res, grad_x_grad_res, negative_slope);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor leaky_relu_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float negative_slope) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::leaky_relu_grad(x_res, out_grad_res, negative_slope);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> lerp_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::lerp_grad(x_res, y_res, weight_res, out_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor lgamma_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::lgamma_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor linear_interp_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  pir::Value output_grad_res = std::static_pointer_cast<LazyTensor>(output_grad.impl())->value();
  auto op_res = paddle::dialect::linear_interp_grad(x_res, out_size_res, size_tensor_res, scale_tensor_res, output_grad_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor log10_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log10_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor log1p_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log1p_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor log2_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log2_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> log_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::log_double_grad(x_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor log_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor log_loss_grad<LazyTensor>(const Tensor& input, const Tensor& label, const Tensor& out_grad, float epsilon) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log_loss_grad(input_res, label_res, out_grad_res, epsilon);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor log_softmax_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad, int axis) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log_softmax_grad(out_res, out_grad_res, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor logcumsumexp_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, int axis, bool flatten, bool exclusive, bool reverse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::logcumsumexp_grad(x_res, out_res, out_grad_res, axis, flatten, exclusive, reverse);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor logit_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float eps) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::logit_grad(x_res, out_grad_res, eps);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor logsigmoid_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::logsigmoid_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor lu_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& pivots, const Tensor& out_grad, bool pivot) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value pivots_res = std::static_pointer_cast<LazyTensor>(pivots.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::lu_grad(x_res, out_res, pivots_res, out_grad_res, pivot);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor lu_unpack_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& l, const Tensor& u, const Tensor& pmat, const Tensor& l_grad, const Tensor& u_grad, bool unpack_ludata, bool unpack_pivots) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value l_res = std::static_pointer_cast<LazyTensor>(l.impl())->value();
  pir::Value u_res = std::static_pointer_cast<LazyTensor>(u.impl())->value();
  pir::Value pmat_res = std::static_pointer_cast<LazyTensor>(pmat.impl())->value();
  pir::Value l_grad_res = std::static_pointer_cast<LazyTensor>(l_grad.impl())->value();
  pir::Value u_grad_res = std::static_pointer_cast<LazyTensor>(u_grad.impl())->value();
  auto op_res = paddle::dialect::lu_unpack_grad(x_res, y_res, l_res, u_res, pmat_res, l_grad_res, u_grad_res, unpack_ludata, unpack_pivots);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor margin_cross_entropy_grad<LazyTensor>(const Tensor& logits, const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale) {
  pir::Value logits_res = std::static_pointer_cast<LazyTensor>(logits.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value softmax_res = std::static_pointer_cast<LazyTensor>(softmax.impl())->value();
  pir::Value loss_grad_res = std::static_pointer_cast<LazyTensor>(loss_grad.impl())->value();
  auto op_res = paddle::dialect::margin_cross_entropy_grad(logits_res, label_res, softmax_res, loss_grad_res, return_softmax, ring_id, rank, nranks, margin1, margin2, margin3, scale);
  Tensor logits_grad(std::make_shared<LazyTensor>(op_res));
  return logits_grad; 
}

template <>
Tensor masked_select_grad<LazyTensor>(const Tensor& x, const Tensor& mask, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value mask_res = std::static_pointer_cast<LazyTensor>(mask.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::masked_select_grad(x_res, mask_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor matrix_power_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, int n) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::matrix_power_grad(x_res, out_res, out_grad_res, n);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor max_pool2d_with_index_grad<LazyTensor>(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value mask_res = std::static_pointer_cast<LazyTensor>(mask.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::max_pool2d_with_index_grad(x_res, mask_res, out_grad_res, kernel_size, strides, paddings, global_pooling, adaptive);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor max_pool3d_with_index_grad<LazyTensor>(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value mask_res = std::static_pointer_cast<LazyTensor>(mask.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::max_pool3d_with_index_grad(x_res, mask_res, out_grad_res, kernel_size, strides, paddings, global_pooling, adaptive);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor maxout_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, int groups, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::maxout_grad(x_res, out_res, out_grad_res, groups, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor mean_all_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::mean_all_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor> memory_efficient_attention_grad<LazyTensor>(const Tensor& query, const Tensor& key, const Tensor& value, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlens_q, const paddle::optional<Tensor>& cu_seqlens_k, const Tensor& output, const Tensor& logsumexp, const Tensor& seed_and_offset, const Tensor& output_grad, const Scalar& max_seqlen_q, const Scalar& max_seqlen_k, bool causal, double dropout_p, float scale) {
  pir::Value query_res = std::static_pointer_cast<LazyTensor>(query.impl())->value();
  pir::Value key_res = std::static_pointer_cast<LazyTensor>(key.impl())->value();
  pir::Value value_res = std::static_pointer_cast<LazyTensor>(value.impl())->value();
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  paddle::optional<pir::Value> cu_seqlens_q_res;
  if(cu_seqlens_q) {
    pir::Value cu_seqlens_q_res_inner;
    cu_seqlens_q_res_inner = std::static_pointer_cast<LazyTensor>(cu_seqlens_q.get().impl())->value();
    cu_seqlens_q_res = paddle::make_optional<pir::Value>(cu_seqlens_q_res_inner);
  }
  paddle::optional<pir::Value> cu_seqlens_k_res;
  if(cu_seqlens_k) {
    pir::Value cu_seqlens_k_res_inner;
    cu_seqlens_k_res_inner = std::static_pointer_cast<LazyTensor>(cu_seqlens_k.get().impl())->value();
    cu_seqlens_k_res = paddle::make_optional<pir::Value>(cu_seqlens_k_res_inner);
  }
  pir::Value output_res = std::static_pointer_cast<LazyTensor>(output.impl())->value();
  pir::Value logsumexp_res = std::static_pointer_cast<LazyTensor>(logsumexp.impl())->value();
  pir::Value seed_and_offset_res = std::static_pointer_cast<LazyTensor>(seed_and_offset.impl())->value();
  pir::Value output_grad_res = std::static_pointer_cast<LazyTensor>(output_grad.impl())->value();
  auto op_res = paddle::dialect::memory_efficient_attention_grad(query_res, key_res, value_res, bias_res, cu_seqlens_q_res, cu_seqlens_k_res, output_res, logsumexp_res, seed_and_offset_res, output_grad_res, max_seqlen_q.to<float>(), max_seqlen_k.to<float>(), causal, dropout_p, scale);
  auto op_res_0 = std::get<0>(op_res);
  Tensor query_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor key_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor value_grad(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(query_grad, key_grad, value_grad, bias_grad); 
}

template <>
std::vector<Tensor> meshgrid_grad<LazyTensor>(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs_grad) {
  std::vector<pir::Value> inputs_res(inputs.size());
  std::transform(inputs.begin(), inputs.end(), inputs_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> outputs_grad_res(outputs_grad.size());
  std::transform(outputs_grad.begin(), outputs_grad.end(), outputs_grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::meshgrid_grad(inputs_res, outputs_grad_res);
  std::vector<Tensor> inputs_grad(op_res.size());
  std::transform(op_res.begin(), op_res.end(), inputs_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return inputs_grad; 
}

template <>
Tensor mode_grad<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::mode_grad(x_res, indices_res, out_grad_res, axis, keepdim);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::vector<Tensor> multi_dot_grad<LazyTensor>(const std::vector<Tensor>& x, const Tensor& out_grad) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::multi_dot_grad(x_res, out_grad_res);
  std::vector<Tensor> x_grad(op_res.size());
  std::transform(op_res.begin(), op_res.end(), x_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return x_grad; 
}

template <>
std::vector<Tensor> multiplex_grad<LazyTensor>(const std::vector<Tensor>& inputs, const Tensor& index, const Tensor& out_grad) {
  std::vector<pir::Value> inputs_res(inputs.size());
  std::transform(inputs.begin(), inputs.end(), inputs_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::multiplex_grad(inputs_res, index_res, out_grad_res);
  std::vector<Tensor> inputs_grad(op_res.size());
  std::transform(op_res.begin(), op_res.end(), inputs_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return inputs_grad; 
}

template <>
std::tuple<Tensor, Tensor> mv_grad<LazyTensor>(const Tensor& x, const Tensor& vec, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value vec_res = std::static_pointer_cast<LazyTensor>(vec.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::mv_grad(x_res, vec_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor vec_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, vec_grad); 
}

template <>
Tensor nanmedian_grad<LazyTensor>(const Tensor& x, const Tensor& medians, const Tensor& out_grad, const IntArray& axis, bool keepdim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value medians_res = std::static_pointer_cast<LazyTensor>(medians.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::nanmedian_grad(x_res, medians_res, out_grad_res, axis.GetData(), keepdim);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor nearest_interp_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  pir::Value output_grad_res = std::static_pointer_cast<LazyTensor>(output_grad.impl())->value();
  auto op_res = paddle::dialect::nearest_interp_grad(x_res, out_size_res, size_tensor_res, scale_tensor_res, output_grad_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor nll_loss_grad<LazyTensor>(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, const Tensor& total_weight, const Tensor& out_grad, int64_t ignore_index, const std::string& reduction) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  paddle::optional<pir::Value> weight_res;
  if(weight) {
    pir::Value weight_res_inner;
    weight_res_inner = std::static_pointer_cast<LazyTensor>(weight.get().impl())->value();
    weight_res = paddle::make_optional<pir::Value>(weight_res_inner);
  }
  pir::Value total_weight_res = std::static_pointer_cast<LazyTensor>(total_weight.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::nll_loss_grad(input_res, label_res, weight_res, total_weight_res, out_grad_res, ignore_index, reduction);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor overlap_add_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, int hop_length, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::overlap_add_grad(x_res, out_grad_res, hop_length, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor p_norm_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, float porder, int axis, float epsilon, bool keepdim, bool asvector) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::p_norm_grad(x_res, out_res, out_grad_res, porder, axis, epsilon, keepdim, asvector);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor pad3d_double_grad<LazyTensor>(const Tensor& grad_x_grad, const IntArray& paddings, const std::string& mode, float pad_value, const std::string& data_format) {
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::pad3d_double_grad(grad_x_grad_res, paddings.GetData(), mode, pad_value, data_format);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor pad3d_double_grad<LazyTensor>(const Tensor& grad_x_grad, const Tensor& paddings_, const std::string& mode, float pad_value, const std::string& data_format) {
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  pir::OpResult paddings_res = std::static_pointer_cast<LazyTensor>(paddings_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::pad3d_double_grad(grad_x_grad_res, paddings_res, mode, pad_value, data_format);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor pad3d_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const IntArray& paddings, const std::string& mode, float pad_value, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::pad3d_grad(x_res, out_grad_res, paddings.GetData(), mode, pad_value, data_format);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor pad3d_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& paddings_, const std::string& mode, float pad_value, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult paddings_res = std::static_pointer_cast<LazyTensor>(paddings_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::pad3d_grad(x_res, out_grad_res, paddings_res, mode, pad_value, data_format);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor pixel_shuffle_grad<LazyTensor>(const Tensor& out_grad, int upscale_factor, const std::string& data_format) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::pixel_shuffle_grad(out_grad_res, upscale_factor, data_format);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor pixel_unshuffle_grad<LazyTensor>(const Tensor& out_grad, int downscale_factor, const std::string& data_format) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::pixel_unshuffle_grad(out_grad_res, downscale_factor, data_format);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor poisson_grad<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::poisson_grad(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor polygamma_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, int n) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::polygamma_grad(x_res, out_grad_res, n);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> pow_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const Scalar& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::pow_double_grad(x_res, grad_out_res, grad_x_grad_res, y.to<float>());
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
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
std::tuple<Tensor, Tensor, Tensor> pow_triple_grad<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_grad_x, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const Scalar& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_grad_x_res = std::static_pointer_cast<LazyTensor>(grad_grad_x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  paddle::optional<pir::Value> grad_grad_out_grad_res;
  if(grad_grad_out_grad) {
    pir::Value grad_grad_out_grad_res_inner;
    grad_grad_out_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_grad_out_grad.get().impl())->value();
    grad_grad_out_grad_res = paddle::make_optional<pir::Value>(grad_grad_out_grad_res_inner);
  }
  auto op_res = paddle::dialect::pow_triple_grad(x_res, grad_out_res, grad_grad_x_res, grad_x_grad_res, grad_grad_out_grad_res, y.to<float>());
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_grad_x_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, grad_out_grad, grad_grad_x_grad); 
}

template <>
std::tuple<Tensor, Tensor> prelu_grad<LazyTensor>(const Tensor& x, const Tensor& alpha, const Tensor& out_grad, const std::string& data_format, const std::string& mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value alpha_res = std::static_pointer_cast<LazyTensor>(alpha.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::prelu_grad(x_res, alpha_res, out_grad_res, data_format, mode);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor alpha_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, alpha_grad); 
}

template <>
Tensor psroi_pool_grad<LazyTensor>(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, int output_channels, float spatial_scale) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value boxes_res = std::static_pointer_cast<LazyTensor>(boxes.impl())->value();
  paddle::optional<pir::Value> boxes_num_res;
  if(boxes_num) {
    pir::Value boxes_num_res_inner;
    boxes_num_res_inner = std::static_pointer_cast<LazyTensor>(boxes_num.get().impl())->value();
    boxes_num_res = paddle::make_optional<pir::Value>(boxes_num_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::psroi_pool_grad(x_res, boxes_res, boxes_num_res, out_grad_res, pooled_height, pooled_width, output_channels, spatial_scale);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> put_along_axis_grad<LazyTensor>(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis, const std::string& reduce) {
  pir::Value arr_res = std::static_pointer_cast<LazyTensor>(arr.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::put_along_axis_grad(arr_res, indices_res, out_grad_res, axis, reduce);
  auto op_res_0 = std::get<0>(op_res);
  Tensor arr_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor values_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(arr_grad, values_grad); 
}

template <>
Tensor qr_grad<LazyTensor>(const Tensor& x, const Tensor& q, const Tensor& r, const Tensor& q_grad, const Tensor& r_grad, const std::string& mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value q_res = std::static_pointer_cast<LazyTensor>(q.impl())->value();
  pir::Value r_res = std::static_pointer_cast<LazyTensor>(r.impl())->value();
  pir::Value q_grad_res = std::static_pointer_cast<LazyTensor>(q_grad.impl())->value();
  pir::Value r_grad_res = std::static_pointer_cast<LazyTensor>(r_grad.impl())->value();
  auto op_res = paddle::dialect::qr_grad(x_res, q_res, r_res, q_grad_res, r_grad_res, mode);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor real_grad<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::real_grad(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor reciprocal_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::reciprocal_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor relu6_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::relu6_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor relu_double_grad<LazyTensor>(const Tensor& out, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::relu_double_grad(out_res, grad_x_grad_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor relu_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::relu_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor renorm_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float p, int axis, float max_norm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::renorm_grad(x_res, out_grad_res, p, axis, max_norm);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor roi_align_grad<LazyTensor>(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value boxes_res = std::static_pointer_cast<LazyTensor>(boxes.impl())->value();
  paddle::optional<pir::Value> boxes_num_res;
  if(boxes_num) {
    pir::Value boxes_num_res_inner;
    boxes_num_res_inner = std::static_pointer_cast<LazyTensor>(boxes_num.get().impl())->value();
    boxes_num_res = paddle::make_optional<pir::Value>(boxes_num_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::roi_align_grad(x_res, boxes_res, boxes_num_res, out_grad_res, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor roi_pool_grad<LazyTensor>(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& arg_max, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value boxes_res = std::static_pointer_cast<LazyTensor>(boxes.impl())->value();
  paddle::optional<pir::Value> boxes_num_res;
  if(boxes_num) {
    pir::Value boxes_num_res_inner;
    boxes_num_res_inner = std::static_pointer_cast<LazyTensor>(boxes_num.get().impl())->value();
    boxes_num_res = paddle::make_optional<pir::Value>(boxes_num_res_inner);
  }
  pir::Value arg_max_res = std::static_pointer_cast<LazyTensor>(arg_max.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::roi_pool_grad(x_res, boxes_res, boxes_num_res, arg_max_res, out_grad_res, pooled_height, pooled_width, spatial_scale);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor roll_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const IntArray& shifts, const std::vector<int64_t>& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::roll_grad(x_res, out_grad_res, shifts.GetData(), axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor roll_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& shifts_, const std::vector<int64_t>& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult shifts_res = std::static_pointer_cast<LazyTensor>(shifts_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::roll_grad(x_res, out_grad_res, shifts_res, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor round_grad<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::round_grad(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> rsqrt_double_grad<LazyTensor>(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_x_res = std::static_pointer_cast<LazyTensor>(grad_x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::rsqrt_double_grad(out_res, grad_x_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_grad, grad_out_grad); 
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
std::tuple<Tensor, Tensor> scatter_grad<LazyTensor>(const Tensor& index, const Tensor& updates, const Tensor& out_grad, bool overwrite) {
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value updates_res = std::static_pointer_cast<LazyTensor>(updates.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::scatter_grad(index_res, updates_res, out_grad_res, overwrite);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor updates_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, updates_grad); 
}

template <>
std::tuple<Tensor, Tensor> scatter_nd_add_grad<LazyTensor>(const Tensor& index, const Tensor& updates, const Tensor& out_grad) {
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value updates_res = std::static_pointer_cast<LazyTensor>(updates.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::scatter_nd_add_grad(index_res, updates_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor updates_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, updates_grad); 
}

template <>
Tensor segment_pool_grad<LazyTensor>(const Tensor& x, const Tensor& segment_ids, const Tensor& out, const paddle::optional<Tensor>& summed_ids, const Tensor& out_grad, const std::string& pooltype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value segment_ids_res = std::static_pointer_cast<LazyTensor>(segment_ids.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  paddle::optional<pir::Value> summed_ids_res;
  if(summed_ids) {
    pir::Value summed_ids_res_inner;
    summed_ids_res_inner = std::static_pointer_cast<LazyTensor>(summed_ids.get().impl())->value();
    summed_ids_res = paddle::make_optional<pir::Value>(summed_ids_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::segment_pool_grad(x_res, segment_ids_res, out_res, summed_ids_res, out_grad_res, pooltype);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor selu_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad, float scale, float alpha) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::selu_grad(out_res, out_grad_res, scale, alpha);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor send_u_recv_grad<LazyTensor>(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& reduce_op) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value src_index_res = std::static_pointer_cast<LazyTensor>(src_index.impl())->value();
  pir::Value dst_index_res = std::static_pointer_cast<LazyTensor>(dst_index.impl())->value();
  paddle::optional<pir::Value> out_res;
  if(out) {
    pir::Value out_res_inner;
    out_res_inner = std::static_pointer_cast<LazyTensor>(out.get().impl())->value();
    out_res = paddle::make_optional<pir::Value>(out_res_inner);
  }
  paddle::optional<pir::Value> dst_count_res;
  if(dst_count) {
    pir::Value dst_count_res_inner;
    dst_count_res_inner = std::static_pointer_cast<LazyTensor>(dst_count.get().impl())->value();
    dst_count_res = paddle::make_optional<pir::Value>(dst_count_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::send_u_recv_grad(x_res, src_index_res, dst_index_res, out_res, dst_count_res, out_grad_res, reduce_op);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> send_ue_recv_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& message_op, const std::string& reduce_op) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value src_index_res = std::static_pointer_cast<LazyTensor>(src_index.impl())->value();
  pir::Value dst_index_res = std::static_pointer_cast<LazyTensor>(dst_index.impl())->value();
  paddle::optional<pir::Value> out_res;
  if(out) {
    pir::Value out_res_inner;
    out_res_inner = std::static_pointer_cast<LazyTensor>(out.get().impl())->value();
    out_res = paddle::make_optional<pir::Value>(out_res_inner);
  }
  paddle::optional<pir::Value> dst_count_res;
  if(dst_count) {
    pir::Value dst_count_res_inner;
    dst_count_res_inner = std::static_pointer_cast<LazyTensor>(dst_count.get().impl())->value();
    dst_count_res = paddle::make_optional<pir::Value>(dst_count_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::send_ue_recv_grad(x_res, y_res, src_index_res, dst_index_res, out_res, dst_count_res, out_grad_res, message_op, reduce_op);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
std::tuple<Tensor, Tensor> send_uv_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const Tensor& out_grad, const std::string& message_op) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value src_index_res = std::static_pointer_cast<LazyTensor>(src_index.impl())->value();
  pir::Value dst_index_res = std::static_pointer_cast<LazyTensor>(dst_index.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::send_uv_grad(x_res, y_res, src_index_res, dst_index_res, out_grad_res, message_op);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor sigmoid_cross_entropy_with_logits_grad<LazyTensor>(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, const Tensor& out_grad, bool normalize, int ignore_index) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  paddle::optional<pir::Value> pos_weight_res;
  if(pos_weight) {
    pir::Value pos_weight_res_inner;
    pos_weight_res_inner = std::static_pointer_cast<LazyTensor>(pos_weight.get().impl())->value();
    pos_weight_res = paddle::make_optional<pir::Value>(pos_weight_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sigmoid_cross_entropy_with_logits_grad(x_res, label_res, pos_weight_res, out_grad_res, normalize, ignore_index);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> sigmoid_double_grad<LazyTensor>(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value fwd_grad_out_res = std::static_pointer_cast<LazyTensor>(fwd_grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::sigmoid_double_grad(out_res, fwd_grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor fwd_grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_grad, fwd_grad_out_grad); 
}

template <>
Tensor sigmoid_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sigmoid_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> sigmoid_triple_grad<LazyTensor>(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_grad_x, const Tensor& grad_out_grad, const paddle::optional<Tensor>& grad_grad_out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value fwd_grad_out_res = std::static_pointer_cast<LazyTensor>(fwd_grad_out.impl())->value();
  pir::Value grad_grad_x_res = std::static_pointer_cast<LazyTensor>(grad_grad_x.impl())->value();
  pir::Value grad_out_grad_res = std::static_pointer_cast<LazyTensor>(grad_out_grad.impl())->value();
  paddle::optional<pir::Value> grad_grad_out_grad_res;
  if(grad_grad_out_grad) {
    pir::Value grad_grad_out_grad_res_inner;
    grad_grad_out_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_grad_out_grad.get().impl())->value();
    grad_grad_out_grad_res = paddle::make_optional<pir::Value>(grad_grad_out_grad_res_inner);
  }
  auto op_res = paddle::dialect::sigmoid_triple_grad(out_res, fwd_grad_out_res, grad_grad_x_res, grad_out_grad_res, grad_grad_out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor fwd_grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_grad_x_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out_grad, fwd_grad_out_grad, grad_grad_x_grad); 
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
std::tuple<Tensor, Tensor> sin_double_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> grad_out_res;
  if(grad_out) {
    pir::Value grad_out_res_inner;
    grad_out_res_inner = std::static_pointer_cast<LazyTensor>(grad_out.get().impl())->value();
    grad_out_res = paddle::make_optional<pir::Value>(grad_out_res_inner);
  }
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::sin_double_grad(x_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor sin_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sin_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> sin_triple_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> grad_out_forward_res;
  if(grad_out_forward) {
    pir::Value grad_out_forward_res_inner;
    grad_out_forward_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_forward.get().impl())->value();
    grad_out_forward_res = paddle::make_optional<pir::Value>(grad_out_forward_res_inner);
  }
  paddle::optional<pir::Value> grad_x_grad_forward_res;
  if(grad_x_grad_forward) {
    pir::Value grad_x_grad_forward_res_inner;
    grad_x_grad_forward_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad_forward.get().impl())->value();
    grad_x_grad_forward_res = paddle::make_optional<pir::Value>(grad_x_grad_forward_res_inner);
  }
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  paddle::optional<pir::Value> grad_out_grad_grad_res;
  if(grad_out_grad_grad) {
    pir::Value grad_out_grad_grad_res_inner;
    grad_out_grad_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_grad_grad.get().impl())->value();
    grad_out_grad_grad_res = paddle::make_optional<pir::Value>(grad_out_grad_grad_res_inner);
  }
  auto op_res = paddle::dialect::sin_triple_grad(x_res, grad_out_forward_res, grad_x_grad_forward_res, grad_x_grad_res, grad_out_grad_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_forward_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_x_grad_forward_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, grad_out_forward_grad, grad_x_grad_forward_grad); 
}

template <>
Tensor sinh_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sinh_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor slogdet_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::slogdet_grad(x_res, out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> softplus_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float beta, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::softplus_double_grad(x_res, grad_out_res, grad_x_grad_res, beta, threshold);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor softplus_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float beta, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::softplus_grad(x_res, out_grad_res, beta, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor softshrink_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::softshrink_grad(x_res, out_grad_res, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor softsign_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::softsign_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> solve_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::solve_grad(x_res, y_res, out_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor spectral_norm_grad<LazyTensor>(const Tensor& weight, const Tensor& u, const Tensor& v, const Tensor& out_grad, int dim, int power_iters, float eps) {
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  pir::Value u_res = std::static_pointer_cast<LazyTensor>(u.impl())->value();
  pir::Value v_res = std::static_pointer_cast<LazyTensor>(v.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::spectral_norm_grad(weight_res, u_res, v_res, out_grad_res, dim, power_iters, eps);
  Tensor weight_grad(std::make_shared<LazyTensor>(op_res));
  return weight_grad; 
}

template <>
std::tuple<Tensor, Tensor> sqrt_double_grad<LazyTensor>(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_x_res = std::static_pointer_cast<LazyTensor>(grad_x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::sqrt_double_grad(out_res, grad_x_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_grad, grad_out_grad); 
}

template <>
Tensor sqrt_grad<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sqrt_grad(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> square_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::square_double_grad(x_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
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
Tensor squared_l2_norm_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::squared_l2_norm_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor squeeze_grad<LazyTensor>(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::squeeze_grad(xshape_res, out_grad_res, axis.GetData());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor squeeze_grad<LazyTensor>(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::squeeze_grad(xshape_res, out_grad_res, axis_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::vector<Tensor> stack_grad<LazyTensor>(const std::vector<Tensor>& x, const Tensor& out_grad, int axis) {
  std::vector<pir::Value> x_res(x.size());
  std::transform(x.begin(), x.end(), x_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::stack_grad(x_res, out_grad_res, axis);
  std::vector<Tensor> x_grad(op_res.size());
  std::transform(op_res.begin(), op_res.end(), x_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return x_grad; 
}

template <>
Tensor stanh_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float scale_a, float scale_b) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::stanh_grad(x_res, out_grad_res, scale_a, scale_b);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor svd_grad<LazyTensor>(const Tensor& x, const Tensor& u, const Tensor& vh, const Tensor& s, const paddle::optional<Tensor>& u_grad, const paddle::optional<Tensor>& vh_grad, const paddle::optional<Tensor>& s_grad, bool full_matrices) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value u_res = std::static_pointer_cast<LazyTensor>(u.impl())->value();
  pir::Value vh_res = std::static_pointer_cast<LazyTensor>(vh.impl())->value();
  pir::Value s_res = std::static_pointer_cast<LazyTensor>(s.impl())->value();
  paddle::optional<pir::Value> u_grad_res;
  if(u_grad) {
    pir::Value u_grad_res_inner;
    u_grad_res_inner = std::static_pointer_cast<LazyTensor>(u_grad.get().impl())->value();
    u_grad_res = paddle::make_optional<pir::Value>(u_grad_res_inner);
  }
  paddle::optional<pir::Value> vh_grad_res;
  if(vh_grad) {
    pir::Value vh_grad_res_inner;
    vh_grad_res_inner = std::static_pointer_cast<LazyTensor>(vh_grad.get().impl())->value();
    vh_grad_res = paddle::make_optional<pir::Value>(vh_grad_res_inner);
  }
  paddle::optional<pir::Value> s_grad_res;
  if(s_grad) {
    pir::Value s_grad_res_inner;
    s_grad_res_inner = std::static_pointer_cast<LazyTensor>(s_grad.get().impl())->value();
    s_grad_res = paddle::make_optional<pir::Value>(s_grad_res_inner);
  }
  auto op_res = paddle::dialect::svd_grad(x_res, u_res, vh_res, s_res, u_grad_res, vh_grad_res, s_grad_res, full_matrices);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor take_along_axis_grad<LazyTensor>(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis) {
  pir::Value arr_res = std::static_pointer_cast<LazyTensor>(arr.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::take_along_axis_grad(arr_res, indices_res, out_grad_res, axis);
  Tensor arr_grad(std::make_shared<LazyTensor>(op_res));
  return arr_grad; 
}

template <>
Tensor tan_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::tan_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> tanh_double_grad<LazyTensor>(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::tanh_double_grad(out_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_grad, grad_out_grad); 
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
Tensor tanh_shrink_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::tanh_shrink_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> tanh_triple_grad<LazyTensor>(const Tensor& out, const Tensor& grad_out_forward, const Tensor& grad_x_grad_forward, const paddle::optional<Tensor>& grad_out_new_grad, const paddle::optional<Tensor>& grad_out_grad_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_out_forward_res = std::static_pointer_cast<LazyTensor>(grad_out_forward.impl())->value();
  pir::Value grad_x_grad_forward_res = std::static_pointer_cast<LazyTensor>(grad_x_grad_forward.impl())->value();
  paddle::optional<pir::Value> grad_out_new_grad_res;
  if(grad_out_new_grad) {
    pir::Value grad_out_new_grad_res_inner;
    grad_out_new_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_new_grad.get().impl())->value();
    grad_out_new_grad_res = paddle::make_optional<pir::Value>(grad_out_new_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_out_grad_grad_res;
  if(grad_out_grad_grad) {
    pir::Value grad_out_grad_grad_res_inner;
    grad_out_grad_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_grad_grad.get().impl())->value();
    grad_out_grad_grad_res = paddle::make_optional<pir::Value>(grad_out_grad_grad_res_inner);
  }
  auto op_res = paddle::dialect::tanh_triple_grad(out_res, grad_out_forward_res, grad_x_grad_forward_res, grad_out_new_grad_res, grad_out_grad_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_forward_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_x_grad_forward_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out_grad, grad_out_forward_grad, grad_x_grad_forward_grad); 
}

template <>
Tensor temporal_shift_grad<LazyTensor>(const Tensor& out_grad, int seg_num, float shift_ratio, const std::string& data_format) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::temporal_shift_grad(out_grad_res, seg_num, shift_ratio, data_format);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor tensor_unfold_grad<LazyTensor>(const Tensor& input, const Tensor& out_grad, int64_t axis, int64_t size, int64_t step) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::tensor_unfold_grad(input_res, out_grad_res, axis, size, step);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor thresholded_relu_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::thresholded_relu_grad(x_res, out_grad_res, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor topk_grad<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& out_grad, const Scalar& k, int axis, bool largest, bool sorted) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::topk_grad(x_res, indices_res, out_grad_res, k.to<int>(), axis, largest, sorted);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor topk_grad<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& out_grad, const Tensor& k_, int axis, bool largest, bool sorted) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult k_res = std::static_pointer_cast<LazyTensor>(k_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::topk_grad(x_res, indices_res, out_grad_res, k_res, axis, largest, sorted);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor trace_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, int offset, int axis1, int axis2) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::trace_grad(x_res, out_grad_res, offset, axis1, axis2);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> triangular_solve_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper, bool transpose, bool unitriangular) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::triangular_solve_grad(x_res, y_res, out_res, out_grad_res, upper, transpose, unitriangular);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor trilinear_interp_grad<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> out_size_res;
  if(out_size) {
    pir::Value out_size_res_inner;
    out_size_res_inner = std::static_pointer_cast<LazyTensor>(out_size.get().impl())->value();
    out_size_res = paddle::make_optional<pir::Value>(out_size_res_inner);
  }
  paddle::optional<std::vector<pir::Value>> size_tensor_res;
  if(size_tensor) {
    std::vector<pir::Value> size_tensor_res_inner(size_tensor.get().size());
    std::transform(size_tensor.get().begin(), size_tensor.get().end(), size_tensor_res_inner.begin(), [](const Tensor& t) {
      return std::static_pointer_cast<LazyTensor>(t.impl())->value();
    });
    size_tensor_res = paddle::make_optional<std::vector<pir::Value>>(size_tensor_res_inner);
  }
  paddle::optional<pir::Value> scale_tensor_res;
  if(scale_tensor) {
    pir::Value scale_tensor_res_inner;
    scale_tensor_res_inner = std::static_pointer_cast<LazyTensor>(scale_tensor.get().impl())->value();
    scale_tensor_res = paddle::make_optional<pir::Value>(scale_tensor_res_inner);
  }
  pir::Value output_grad_res = std::static_pointer_cast<LazyTensor>(output_grad.impl())->value();
  auto op_res = paddle::dialect::trilinear_interp_grad(x_res, out_size_res, size_tensor_res, scale_tensor_res, output_grad_res, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor trunc_grad<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::trunc_grad(out_grad_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor unfold_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::unfold_grad(x_res, out_grad_res, kernel_sizes, strides, paddings, dilations);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor uniform_inplace_grad<LazyTensor>(const Tensor& out_grad, float min, float max, int seed, int diag_num, int diag_step, float diag_val) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::uniform_inplace_grad(out_grad_res, min, max, seed, diag_num, diag_step, diag_val);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor unsqueeze_grad<LazyTensor>(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::unsqueeze_grad(xshape_res, out_grad_res, axis.GetData());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor unsqueeze_grad<LazyTensor>(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::unsqueeze_grad(xshape_res, out_grad_res, axis_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor unstack_grad<LazyTensor>(const std::vector<Tensor>& out_grad, int axis) {
  std::vector<pir::Value> out_grad_res(out_grad.size());
  std::transform(out_grad.begin(), out_grad.end(), out_grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::unstack_grad(out_grad_res, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor view_dtype_grad<LazyTensor>(const Tensor& input, const Tensor& out_grad, DataType dtype) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::view_dtype_grad(input_res, out_grad_res, dtype);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor view_shape_grad<LazyTensor>(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::view_shape_grad(input_res, out_grad_res, dims);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor warpctc_grad<LazyTensor>(const Tensor& logits, const paddle::optional<Tensor>& logits_length, const Tensor& warpctcgrad, const Tensor& loss_grad, int blank, bool norm_by_times) {
  pir::Value logits_res = std::static_pointer_cast<LazyTensor>(logits.impl())->value();
  paddle::optional<pir::Value> logits_length_res;
  if(logits_length) {
    pir::Value logits_length_res_inner;
    logits_length_res_inner = std::static_pointer_cast<LazyTensor>(logits_length.get().impl())->value();
    logits_length_res = paddle::make_optional<pir::Value>(logits_length_res_inner);
  }
  pir::Value warpctcgrad_res = std::static_pointer_cast<LazyTensor>(warpctcgrad.impl())->value();
  pir::Value loss_grad_res = std::static_pointer_cast<LazyTensor>(loss_grad.impl())->value();
  auto op_res = paddle::dialect::warpctc_grad(logits_res, logits_length_res, warpctcgrad_res, loss_grad_res, blank, norm_by_times);
  Tensor logits_grad(std::make_shared<LazyTensor>(op_res));
  return logits_grad; 
}

template <>
Tensor warprnnt_grad<LazyTensor>(const Tensor& input, const Tensor& input_lengths, const Tensor& warprnntgrad, const Tensor& loss_grad, int blank, float fastemit_lambda) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value input_lengths_res = std::static_pointer_cast<LazyTensor>(input_lengths.impl())->value();
  pir::Value warprnntgrad_res = std::static_pointer_cast<LazyTensor>(warprnntgrad.impl())->value();
  pir::Value loss_grad_res = std::static_pointer_cast<LazyTensor>(loss_grad.impl())->value();
  auto op_res = paddle::dialect::warprnnt_grad(input_res, input_lengths_res, warprnntgrad_res, loss_grad_res, blank, fastemit_lambda);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor weight_only_linear_grad<LazyTensor>(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, const Tensor& out_grad, const std::string& weight_dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  pir::Value weight_scale_res = std::static_pointer_cast<LazyTensor>(weight_scale.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::weight_only_linear_grad(x_res, weight_res, bias_res, weight_scale_res, out_grad_res, weight_dtype);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> where_grad<LazyTensor>(const Tensor& condition, const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value condition_res = std::static_pointer_cast<LazyTensor>(condition.impl())->value();
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::where_grad(condition_res, x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor> yolo_loss_grad<LazyTensor>(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const Tensor& objectness_mask, const Tensor& gt_match_mask, const Tensor& loss_grad, const std::vector<int>& anchors, const std::vector<int>& anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth, float scale_x_y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value gt_box_res = std::static_pointer_cast<LazyTensor>(gt_box.impl())->value();
  pir::Value gt_label_res = std::static_pointer_cast<LazyTensor>(gt_label.impl())->value();
  paddle::optional<pir::Value> gt_score_res;
  if(gt_score) {
    pir::Value gt_score_res_inner;
    gt_score_res_inner = std::static_pointer_cast<LazyTensor>(gt_score.get().impl())->value();
    gt_score_res = paddle::make_optional<pir::Value>(gt_score_res_inner);
  }
  pir::Value objectness_mask_res = std::static_pointer_cast<LazyTensor>(objectness_mask.impl())->value();
  pir::Value gt_match_mask_res = std::static_pointer_cast<LazyTensor>(gt_match_mask.impl())->value();
  pir::Value loss_grad_res = std::static_pointer_cast<LazyTensor>(loss_grad.impl())->value();
  auto op_res = paddle::dialect::yolo_loss_grad(x_res, gt_box_res, gt_label_res, gt_score_res, objectness_mask_res, gt_match_mask_res, loss_grad_res, anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, use_label_smooth, scale_x_y);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor gt_box_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor gt_label_grad(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor gt_score_grad(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(x_grad, gt_box_grad, gt_label_grad, gt_score_grad); 
}

template <>
Tensor unpool3d_grad<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_size, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::unpool3d_grad(x_res, indices_res, out_res, out_grad_res, ksize, strides, paddings, output_size, data_format);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor add_double_grad<LazyTensor>(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis) {
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  auto op_res = paddle::dialect::add_double_grad(y_res, grad_out_res, grad_x_grad_res, grad_y_grad_res, axis);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
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
std::tuple<Tensor, Tensor> add_triple_grad<LazyTensor>(const Tensor& grad_grad_x, const Tensor& grad_grad_y, const Tensor& grad_grad_out_grad, int axis) {
  pir::Value grad_grad_x_res = std::static_pointer_cast<LazyTensor>(grad_grad_x.impl())->value();
  pir::Value grad_grad_y_res = std::static_pointer_cast<LazyTensor>(grad_grad_y.impl())->value();
  pir::Value grad_grad_out_grad_res = std::static_pointer_cast<LazyTensor>(grad_grad_out_grad.impl())->value();
  auto op_res = paddle::dialect::add_triple_grad(grad_grad_x_res, grad_grad_y_res, grad_grad_out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor grad_grad_x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_grad_y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(grad_grad_x_grad, grad_grad_y_grad); 
}

template <>
Tensor amax_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::amax_grad(x_res, out_res, out_grad_res, axis, keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor amin_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::amin_grad(x_res, out_res, out_grad_res, axis, keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor assign_out__grad<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::assign_out__grad(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> batch_norm_double_grad<LazyTensor>(const Tensor& x, const Tensor& scale, const paddle::optional<Tensor>& out_mean, const paddle::optional<Tensor>& out_variance, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  paddle::optional<pir::Value> out_mean_res;
  if(out_mean) {
    pir::Value out_mean_res_inner;
    out_mean_res_inner = std::static_pointer_cast<LazyTensor>(out_mean.get().impl())->value();
    out_mean_res = paddle::make_optional<pir::Value>(out_mean_res_inner);
  }
  paddle::optional<pir::Value> out_variance_res;
  if(out_variance) {
    pir::Value out_variance_res_inner;
    out_variance_res_inner = std::static_pointer_cast<LazyTensor>(out_variance.get().impl())->value();
    out_variance_res = paddle::make_optional<pir::Value>(out_variance_res_inner);
  }
  pir::Value saved_mean_res = std::static_pointer_cast<LazyTensor>(saved_mean.impl())->value();
  pir::Value saved_variance_res = std::static_pointer_cast<LazyTensor>(saved_variance.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_scale_grad_res;
  if(grad_scale_grad) {
    pir::Value grad_scale_grad_res_inner;
    grad_scale_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_scale_grad.get().impl())->value();
    grad_scale_grad_res = paddle::make_optional<pir::Value>(grad_scale_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_bias_grad_res;
  if(grad_bias_grad) {
    pir::Value grad_bias_grad_res_inner;
    grad_bias_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_bias_grad.get().impl())->value();
    grad_bias_grad_res = paddle::make_optional<pir::Value>(grad_bias_grad_res_inner);
  }
  auto op_res = paddle::dialect::batch_norm_double_grad(x_res, scale_res, out_mean_res, out_variance_res, saved_mean_res, saved_variance_res, grad_out_res, grad_x_grad_res, grad_scale_grad_res, grad_bias_grad_res, momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, scale_grad, grad_out_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> batch_norm_grad<LazyTensor>(const Tensor& x, const Tensor& scale, const Tensor& bias, const paddle::optional<Tensor>& mean_out, const paddle::optional<Tensor>& variance_out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  pir::Value bias_res = std::static_pointer_cast<LazyTensor>(bias.impl())->value();
  paddle::optional<pir::Value> mean_out_res;
  if(mean_out) {
    pir::Value mean_out_res_inner;
    mean_out_res_inner = std::static_pointer_cast<LazyTensor>(mean_out.get().impl())->value();
    mean_out_res = paddle::make_optional<pir::Value>(mean_out_res_inner);
  }
  paddle::optional<pir::Value> variance_out_res;
  if(variance_out) {
    pir::Value variance_out_res_inner;
    variance_out_res_inner = std::static_pointer_cast<LazyTensor>(variance_out.get().impl())->value();
    variance_out_res = paddle::make_optional<pir::Value>(variance_out_res_inner);
  }
  pir::Value saved_mean_res = std::static_pointer_cast<LazyTensor>(saved_mean.impl())->value();
  pir::Value saved_variance_res = std::static_pointer_cast<LazyTensor>(saved_variance.impl())->value();
  paddle::optional<pir::Value> reserve_space_res;
  if(reserve_space) {
    pir::Value reserve_space_res_inner;
    reserve_space_res_inner = std::static_pointer_cast<LazyTensor>(reserve_space.get().impl())->value();
    reserve_space_res = paddle::make_optional<pir::Value>(reserve_space_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::batch_norm_grad(x_res, scale_res, bias_res, mean_out_res, variance_out_res, saved_mean_res, saved_variance_res, reserve_space_res, out_grad_res, momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, scale_grad, bias_grad); 
}

template <>
Tensor c_embedding_grad<LazyTensor>(const Tensor& weight, const Tensor& x, const Tensor& out_grad, int64_t start_index) {
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::c_embedding_grad(weight_res, x_res, out_grad_res, start_index);
  Tensor weight_grad(std::make_shared<LazyTensor>(op_res));
  return weight_grad; 
}

template <>
Tensor channel_shuffle_grad<LazyTensor>(const Tensor& out_grad, int groups, const std::string& data_format) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::channel_shuffle_grad(out_grad_res, groups, data_format);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> conv2d_transpose_double_grad<LazyTensor>(const Tensor& x, const Tensor& filter, const Tensor& grad_out, const Tensor& grad_x_grad, const Tensor& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  pir::Value grad_filter_grad_res = std::static_pointer_cast<LazyTensor>(grad_filter_grad.impl())->value();
  auto op_res = paddle::dialect::conv2d_transpose_double_grad(x_res, filter_res, grad_out_res, grad_x_grad_res, grad_filter_grad_res, strides, paddings, output_padding, output_size.GetData(), padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, filter_grad, grad_out_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> conv2d_transpose_double_grad<LazyTensor>(const Tensor& x, const Tensor& filter, const Tensor& grad_out, const Tensor& grad_x_grad, const Tensor& grad_filter_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  pir::Value grad_filter_grad_res = std::static_pointer_cast<LazyTensor>(grad_filter_grad.impl())->value();
  pir::OpResult output_size_res = std::static_pointer_cast<LazyTensor>(output_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::conv2d_transpose_double_grad(x_res, filter_res, grad_out_res, grad_x_grad_res, grad_filter_grad_res, output_size_res, strides, paddings, output_padding, padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, filter_grad, grad_out_grad); 
}

template <>
std::tuple<Tensor, Tensor> conv2d_transpose_grad<LazyTensor>(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::conv2d_transpose_grad(x_res, filter_res, out_grad_res, strides, paddings, output_padding, output_size.GetData(), padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, filter_grad); 
}

template <>
std::tuple<Tensor, Tensor> conv2d_transpose_grad<LazyTensor>(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult output_size_res = std::static_pointer_cast<LazyTensor>(output_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::conv2d_transpose_grad(x_res, filter_res, out_grad_res, output_size_res, strides, paddings, output_padding, padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, filter_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor> deformable_conv_grad<LazyTensor>(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value offset_res = std::static_pointer_cast<LazyTensor>(offset.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  paddle::optional<pir::Value> mask_res;
  if(mask) {
    pir::Value mask_res_inner;
    mask_res_inner = std::static_pointer_cast<LazyTensor>(mask.get().impl())->value();
    mask_res = paddle::make_optional<pir::Value>(mask_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::deformable_conv_grad(x_res, offset_res, filter_res, mask_res, out_grad_res, strides, paddings, dilations, deformable_groups, groups, im2col_step);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor offset_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor mask_grad(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(x_grad, offset_grad, filter_grad, mask_grad); 
}

template <>
std::tuple<Tensor, Tensor> depthwise_conv2d_transpose_grad<LazyTensor>(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const IntArray& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::depthwise_conv2d_transpose_grad(x_res, filter_res, out_grad_res, strides, paddings, output_padding, output_size.GetData(), padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, filter_grad); 
}

template <>
std::tuple<Tensor, Tensor> depthwise_conv2d_transpose_grad<LazyTensor>(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value filter_res = std::static_pointer_cast<LazyTensor>(filter.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult output_size_res = std::static_pointer_cast<LazyTensor>(output_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::depthwise_conv2d_transpose_grad(x_res, filter_res, out_grad_res, output_size_res, strides, paddings, output_padding, padding_algorithm, groups, dilations, data_format);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor filter_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, filter_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> divide_double_grad<LazyTensor>(const Tensor& y, const Tensor& out, const Tensor& grad_x, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis) {
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_x_res = std::static_pointer_cast<LazyTensor>(grad_x.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  auto op_res = paddle::dialect::divide_double_grad(y_res, out_res, grad_x_res, grad_x_grad_res, grad_y_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(y_grad, out_grad, grad_out_grad); 
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
std::vector<Tensor> einsum_grad<LazyTensor>(const std::vector<Tensor>& x_shape, const std::vector<Tensor>& inner_cache, const Tensor& out_grad, const std::string& equation) {
  std::vector<pir::Value> x_shape_res(x_shape.size());
  std::transform(x_shape.begin(), x_shape.end(), x_shape_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> inner_cache_res(inner_cache.size());
  std::transform(inner_cache.begin(), inner_cache.end(), inner_cache_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::einsum_grad(x_shape_res, inner_cache_res, out_grad_res, equation);
  std::vector<Tensor> x_grad(op_res.size());
  std::transform(op_res.begin(), op_res.end(), x_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
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
Tensor frobenius_norm_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keep_dim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::frobenius_norm_grad(x_res, out_res, out_grad_res, axis, keep_dim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> fused_batch_norm_act_grad<LazyTensor>(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  pir::Value bias_res = std::static_pointer_cast<LazyTensor>(bias.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value saved_mean_res = std::static_pointer_cast<LazyTensor>(saved_mean.impl())->value();
  pir::Value saved_variance_res = std::static_pointer_cast<LazyTensor>(saved_variance.impl())->value();
  paddle::optional<pir::Value> reserve_space_res;
  if(reserve_space) {
    pir::Value reserve_space_res_inner;
    reserve_space_res_inner = std::static_pointer_cast<LazyTensor>(reserve_space.get().impl())->value();
    reserve_space_res = paddle::make_optional<pir::Value>(reserve_space_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fused_batch_norm_act_grad(x_res, scale_res, bias_res, out_res, saved_mean_res, saved_variance_res, reserve_space_res, out_grad_res, momentum, epsilon, act_type);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, scale_grad, bias_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor, Tensor> fused_bn_add_activation_grad<LazyTensor>(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  pir::Value bias_res = std::static_pointer_cast<LazyTensor>(bias.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value saved_mean_res = std::static_pointer_cast<LazyTensor>(saved_mean.impl())->value();
  pir::Value saved_variance_res = std::static_pointer_cast<LazyTensor>(saved_variance.impl())->value();
  paddle::optional<pir::Value> reserve_space_res;
  if(reserve_space) {
    pir::Value reserve_space_res_inner;
    reserve_space_res_inner = std::static_pointer_cast<LazyTensor>(reserve_space.get().impl())->value();
    reserve_space_res = paddle::make_optional<pir::Value>(reserve_space_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fused_bn_add_activation_grad(x_res, scale_res, bias_res, out_res, saved_mean_res, saved_variance_res, reserve_space_res, out_grad_res, momentum, epsilon, act_type);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor z_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_3));
  return std::make_tuple(x_grad, z_grad, scale_grad, bias_grad); 
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
Tensor hardswish_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::hardswish_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> hsigmoid_loss_grad<LazyTensor>(const Tensor& x, const Tensor& w, const Tensor& label, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, const paddle::optional<Tensor>& bias, const Tensor& pre_out, const Tensor& out_grad, int num_classes, bool is_sparse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value w_res = std::static_pointer_cast<LazyTensor>(w.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  paddle::optional<pir::Value> path_res;
  if(path) {
    pir::Value path_res_inner;
    path_res_inner = std::static_pointer_cast<LazyTensor>(path.get().impl())->value();
    path_res = paddle::make_optional<pir::Value>(path_res_inner);
  }
  paddle::optional<pir::Value> code_res;
  if(code) {
    pir::Value code_res_inner;
    code_res_inner = std::static_pointer_cast<LazyTensor>(code.get().impl())->value();
    code_res = paddle::make_optional<pir::Value>(code_res_inner);
  }
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  pir::Value pre_out_res = std::static_pointer_cast<LazyTensor>(pre_out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::hsigmoid_loss_grad(x_res, w_res, label_res, path_res, code_res, bias_res, pre_out_res, out_grad_res, num_classes, is_sparse);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor w_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, w_grad, bias_grad); 
}

template <>
Tensor logsumexp_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::logsumexp_grad(x_res, out_res, out_grad_res, axis, keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> matmul_double_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, bool transpose_x, bool transpose_y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  auto op_res = paddle::dialect::matmul_double_grad(x_res, y_res, grad_out_res, grad_x_grad_res, grad_y_grad_res, transpose_x, transpose_y);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, y_grad, grad_out_grad); 
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
Tensor max_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::max_grad(x_res, out_res, out_grad_res, axis.GetData(), keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor max_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::max_grad(x_res, out_res, out_grad_res, axis_res, keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> maximum_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::maximum_grad(x_res, y_res, out_grad_res);
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
Tensor min_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::min_grad(x_res, out_res, out_grad_res, axis.GetData(), keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor min_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::min_grad(x_res, out_res, out_grad_res, axis_res, keepdim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> minimum_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::minimum_grad(x_res, y_res, out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor mish_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::mish_grad(x_res, out_grad_res, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> multiply_double_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  auto op_res = paddle::dialect::multiply_double_grad(x_res, y_res, grad_out_res, grad_x_grad_res, grad_y_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, y_grad, grad_out_grad); 
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
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> multiply_triple_grad<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& fwd_grad_out, const paddle::optional<Tensor>& fwd_grad_grad_x, const paddle::optional<Tensor>& fwd_grad_grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, const paddle::optional<Tensor>& grad_grad_out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value fwd_grad_out_res = std::static_pointer_cast<LazyTensor>(fwd_grad_out.impl())->value();
  paddle::optional<pir::Value> fwd_grad_grad_x_res;
  if(fwd_grad_grad_x) {
    pir::Value fwd_grad_grad_x_res_inner;
    fwd_grad_grad_x_res_inner = std::static_pointer_cast<LazyTensor>(fwd_grad_grad_x.get().impl())->value();
    fwd_grad_grad_x_res = paddle::make_optional<pir::Value>(fwd_grad_grad_x_res_inner);
  }
  paddle::optional<pir::Value> fwd_grad_grad_y_res;
  if(fwd_grad_grad_y) {
    pir::Value fwd_grad_grad_y_res_inner;
    fwd_grad_grad_y_res_inner = std::static_pointer_cast<LazyTensor>(fwd_grad_grad_y.get().impl())->value();
    fwd_grad_grad_y_res = paddle::make_optional<pir::Value>(fwd_grad_grad_y_res_inner);
  }
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_grad_out_grad_res;
  if(grad_grad_out_grad) {
    pir::Value grad_grad_out_grad_res_inner;
    grad_grad_out_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_grad_out_grad.get().impl())->value();
    grad_grad_out_grad_res = paddle::make_optional<pir::Value>(grad_grad_out_grad_res_inner);
  }
  auto op_res = paddle::dialect::multiply_triple_grad(x_res, y_res, fwd_grad_out_res, fwd_grad_grad_x_res, fwd_grad_grad_y_res, grad_x_grad_res, grad_y_grad_res, grad_grad_out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor fwd_grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  auto op_res_3 = std::get<3>(op_res);
  Tensor fwd_grad_grad_x_grad(std::make_shared<LazyTensor>(op_res_3));
  auto op_res_4 = std::get<4>(op_res);
  Tensor fwd_grad_grad_y_grad(std::make_shared<LazyTensor>(op_res_4));
  return std::make_tuple(x_grad, y_grad, fwd_grad_out_grad, fwd_grad_grad_x_grad, fwd_grad_grad_y_grad); 
}

template <>
Tensor norm_grad<LazyTensor>(const Tensor& x, const Tensor& norm, const Tensor& out_grad, int axis, float epsilon, bool is_test) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value norm_res = std::static_pointer_cast<LazyTensor>(norm.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::norm_grad(x_res, norm_res, out_grad_res, axis, epsilon, is_test);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor pad_double_grad<LazyTensor>(const Tensor& grad_x_grad, const std::vector<int>& paddings, const Scalar& pad_value) {
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::pad_double_grad(grad_x_grad_res, paddings, pad_value.to<float>());
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor pad_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const std::vector<int>& paddings, const Scalar& pad_value) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::pad_grad(x_res, out_grad_res, paddings, pad_value.to<float>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor pool2d_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_x_grad, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::pool2d_double_grad(x_res, grad_x_grad_res, kernel_size.GetData(), strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor pool2d_double_grad<LazyTensor>(const Tensor& x, const Tensor& grad_x_grad, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  pir::OpResult kernel_size_res = std::static_pointer_cast<LazyTensor>(kernel_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::pool2d_double_grad(x_res, grad_x_grad_res, kernel_size_res, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor pool2d_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::pool2d_grad(x_res, out_res, out_grad_res, kernel_size.GetData(), strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor pool2d_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult kernel_size_res = std::static_pointer_cast<LazyTensor>(kernel_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::pool2d_grad(x_res, out_res, out_grad_res, kernel_size_res, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor pool3d_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::pool3d_grad(x_res, out_res, out_grad_res, kernel_size, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor prod_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const IntArray& dims, bool keep_dim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::prod_grad(x_res, out_res, out_grad_res, dims.GetData(), keep_dim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor prod_grad<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& dims_, bool keep_dim, bool reduce_all) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult dims_res = std::static_pointer_cast<LazyTensor>(dims_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::prod_grad(x_res, out_res, out_grad_res, dims_res, keep_dim, reduce_all);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor repeat_interleave_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, int repeats, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::repeat_interleave_grad(x_res, out_grad_res, repeats, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor repeat_interleave_with_tensor_index_grad<LazyTensor>(const Tensor& x, const Tensor& repeats, const Tensor& out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value repeats_res = std::static_pointer_cast<LazyTensor>(repeats.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::repeat_interleave_with_tensor_index_grad(x_res, repeats_res, out_grad_res, axis);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor reshape_double_grad<LazyTensor>(const Tensor& grad_out, const Tensor& grad_x_grad) {
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::reshape_double_grad(grad_out_res, grad_x_grad_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
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
std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> rnn_grad<LazyTensor>(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& out, const Tensor& dropout_state_out, const Tensor& reserve, const Tensor& out_grad, const std::vector<Tensor>& state_grad, float dropout_prob, bool is_bidirec, int input_size, int hidden_size, int num_layers, const std::string& mode, int seed, bool is_test) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  std::vector<pir::Value> pre_state_res(pre_state.size());
  std::transform(pre_state.begin(), pre_state.end(), pre_state_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  std::vector<pir::Value> weight_list_res(weight_list.size());
  std::transform(weight_list.begin(), weight_list.end(), weight_list_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  paddle::optional<pir::Value> sequence_length_res;
  if(sequence_length) {
    pir::Value sequence_length_res_inner;
    sequence_length_res_inner = std::static_pointer_cast<LazyTensor>(sequence_length.get().impl())->value();
    sequence_length_res = paddle::make_optional<pir::Value>(sequence_length_res_inner);
  }
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value dropout_state_out_res = std::static_pointer_cast<LazyTensor>(dropout_state_out.impl())->value();
  pir::Value reserve_res = std::static_pointer_cast<LazyTensor>(reserve.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  std::vector<pir::Value> state_grad_res(state_grad.size());
  std::transform(state_grad.begin(), state_grad.end(), state_grad_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  auto op_res = paddle::dialect::rnn_grad(x_res, pre_state_res, weight_list_res, sequence_length_res, out_res, dropout_state_out_res, reserve_res, out_grad_res, state_grad_res, dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode, seed, is_test);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  std::vector<Tensor> pre_state_grad(op_res_1.size());
  std::transform(op_res_1.begin(), op_res_1.end(), pre_state_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  auto op_res_2 = std::get<2>(op_res);
  std::vector<Tensor> weight_list_grad(op_res_2.size());
  std::transform(op_res_2.begin(), op_res_2.end(), weight_list_grad.begin(), [](const pir::OpResult& res) {
  return Tensor(std::make_shared<LazyTensor>(res));
    });
  return std::make_tuple(x_grad, pre_state_grad, weight_list_grad); 
}

template <>
Tensor rrelu_grad<LazyTensor>(const Tensor& x, const Tensor& noise, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value noise_res = std::static_pointer_cast<LazyTensor>(noise.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::rrelu_grad(x_res, noise_res, out_grad_res);
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
Tensor strided_slice_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const std::vector<int>& axes, const IntArray& starts, const IntArray& ends, const IntArray& strides) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::strided_slice_grad(x_res, out_grad_res, axes, starts.GetData(), ends.GetData(), strides.GetData());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor strided_slice_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const Tensor& strides_, const std::vector<int>& axes) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult starts_res = std::static_pointer_cast<LazyTensor>(starts_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult ends_res = std::static_pointer_cast<LazyTensor>(ends_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult strides_res = std::static_pointer_cast<LazyTensor>(strides_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::strided_slice_grad(x_res, out_grad_res, starts_res, ends_res, strides_res, axes);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor subtract_double_grad<LazyTensor>(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis) {
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  auto op_res = paddle::dialect::subtract_double_grad(y_res, grad_out_res, grad_x_grad_res, grad_y_grad_res, axis);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
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
Tensor swish_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::swish_grad(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> sync_batch_norm_grad<LazyTensor>(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  pir::Value bias_res = std::static_pointer_cast<LazyTensor>(bias.impl())->value();
  pir::Value saved_mean_res = std::static_pointer_cast<LazyTensor>(saved_mean.impl())->value();
  pir::Value saved_variance_res = std::static_pointer_cast<LazyTensor>(saved_variance.impl())->value();
  paddle::optional<pir::Value> reserve_space_res;
  if(reserve_space) {
    pir::Value reserve_space_res_inner;
    reserve_space_res_inner = std::static_pointer_cast<LazyTensor>(reserve_space.get().impl())->value();
    reserve_space_res = paddle::make_optional<pir::Value>(reserve_space_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sync_batch_norm_grad(x_res, scale_res, bias_res, saved_mean_res, saved_variance_res, reserve_space_res, out_grad_res, momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, scale_grad, bias_grad); 
}

template <>
Tensor tile_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const IntArray& repeat_times) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::tile_grad(x_res, out_grad_res, repeat_times.GetData());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor tile_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& repeat_times_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult repeat_times_res = std::static_pointer_cast<LazyTensor>(repeat_times_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::tile_grad(x_res, out_grad_res, repeat_times_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor trans_layout_grad<LazyTensor>(const Tensor& x, const Tensor& out_grad, const std::vector<int>& perm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::trans_layout_grad(x_res, out_grad_res, perm);
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

template <>
Tensor tril_grad<LazyTensor>(const Tensor& out_grad, int diagonal) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::tril_grad(out_grad_res, diagonal);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor triu_grad<LazyTensor>(const Tensor& out_grad, int diagonal) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::triu_grad(out_grad_res, diagonal);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor disable_check_model_nan_inf_grad<LazyTensor>(const Tensor& out_grad, int unsetflag) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::disable_check_model_nan_inf_grad(out_grad_res, unsetflag);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor enable_check_model_nan_inf_grad<LazyTensor>(const Tensor& out_grad, int unsetflag) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::enable_check_model_nan_inf_grad(out_grad_res, unsetflag);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor unpool_grad<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const IntArray& output_size, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::unpool_grad(x_res, indices_res, out_res, out_grad_res, ksize, strides, padding, output_size.GetData(), data_format);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor unpool_grad<LazyTensor>(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const std::string& data_format) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult output_size_res = std::static_pointer_cast<LazyTensor>(output_size_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::unpool_grad(x_res, indices_res, out_res, out_grad_res, output_size_res, ksize, strides, padding, data_format);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor abs_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::abs_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor acos_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::acos_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor acosh_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::acosh_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor addmm_<LazyTensor>(const Tensor& input, const Tensor& x, const Tensor& y, float beta, float alpha) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::addmm_(input_res, x_res, y_res, beta, alpha);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor asin_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::asin_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor asinh_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::asinh_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor atan_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::atan_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor atanh_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::atanh_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bce_loss_<LazyTensor>(const Tensor& input, const Tensor& label) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::bce_loss_(input_res, label_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bitwise_and_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::bitwise_and_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bitwise_not_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::bitwise_not_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bitwise_or_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::bitwise_or_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor bitwise_xor_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::bitwise_xor_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor ceil_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::ceil_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor clip_<LazyTensor>(const Tensor& x, const Scalar& min, const Scalar& max) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::clip_(x_res, min.to<float>(), max.to<float>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor clip_<LazyTensor>(const Tensor& x, const Tensor& min_, const Tensor& max_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult min_res = std::static_pointer_cast<LazyTensor>(min_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult max_res = std::static_pointer_cast<LazyTensor>(max_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::clip_(x_res, min_res, max_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cos_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cos_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cosh_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cosh_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor> cross_entropy_with_softmax_<LazyTensor>(const Tensor& input, const Tensor& label, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  auto op_res = paddle::dialect::cross_entropy_with_softmax_(input_res, label_res, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor softmax(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor loss(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(softmax, loss); 
}

template <>
Tensor cumprod_<LazyTensor>(const Tensor& x, int dim) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cumprod_(x_res, dim);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cumsum_<LazyTensor>(const Tensor& x, const Scalar& axis, bool flatten, bool exclusive, bool reverse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cumsum_(x_res, axis.to<int>(), flatten, exclusive, reverse);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cumsum_<LazyTensor>(const Tensor& x, const Tensor& axis_, bool flatten, bool exclusive, bool reverse) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::cumsum_(x_res, axis_res, flatten, exclusive, reverse);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor digamma_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::digamma_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor elu_<LazyTensor>(const Tensor& x, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::elu_(x_res, alpha);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor erf_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::erf_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor erfinv_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::erfinv_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor exp_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::exp_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor expm1_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::expm1_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fill_<LazyTensor>(const Tensor& x, const Scalar& value) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::fill_(x_res, value.to<float>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fill_<LazyTensor>(const Tensor& x, const Tensor& value_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult value_res = std::static_pointer_cast<LazyTensor>(value_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::fill_(x_res, value_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fill_diagonal_<LazyTensor>(const Tensor& x, float value, int offset, bool wrap) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::fill_diagonal_(x_res, value, offset, wrap);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor fill_diagonal_tensor_<LazyTensor>(const Tensor& x, const Tensor& y, int64_t offset, int dim1, int dim2) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::fill_diagonal_tensor_(x_res, y_res, offset, dim1, dim2);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor flatten_<LazyTensor>(const Tensor& x, int start_axis, int stop_axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::flatten_(x_res, start_axis, stop_axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor floor_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::floor_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor gaussian_inplace_<LazyTensor>(const Tensor& x, float mean, float std, int seed) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::gaussian_inplace_(x_res, mean, std, seed);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor hardtanh_<LazyTensor>(const Tensor& x, float t_min, float t_max) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::hardtanh_(x_res, t_min, t_max);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor i0_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::i0_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor index_add_<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& add_value, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value add_value_res = std::static_pointer_cast<LazyTensor>(add_value.impl())->value();
  auto op_res = paddle::dialect::index_add_(x_res, index_res, add_value_res, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor index_put_<LazyTensor>(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, bool accumulate) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  std::vector<pir::Value> indices_res(indices.size());
  std::transform(indices.begin(), indices.end(), indices_res.begin(), [](const Tensor& t) {
    return std::static_pointer_cast<LazyTensor>(t.impl())->value();
  });
  pir::Value value_res = std::static_pointer_cast<LazyTensor>(value.impl())->value();
  auto op_res = paddle::dialect::index_put_(x_res, indices_res, value_res, accumulate);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor leaky_relu_<LazyTensor>(const Tensor& x, float negative_slope) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::leaky_relu_(x_res, negative_slope);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor lerp_<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& weight) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value weight_res = std::static_pointer_cast<LazyTensor>(weight.impl())->value();
  auto op_res = paddle::dialect::lerp_(x_res, y_res, weight_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor lgamma_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::lgamma_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor log_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::log_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor log10_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::log10_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor log1p_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::log1p_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor log2_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::log2_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logical_and_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::logical_and_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logical_not_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::logical_not_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logical_or_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::logical_or_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logical_xor_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::logical_xor_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor logit_<LazyTensor>(const Tensor& x, float eps) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::logit_(x_res, eps);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> lu_<LazyTensor>(const Tensor& x, bool pivot) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::lu_(x_res, pivot);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor pivots(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor infos(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out, pivots, infos); 
}

template <>
Tensor polygamma_<LazyTensor>(const Tensor& x, int n) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::polygamma_(x_res, n);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor pow_<LazyTensor>(const Tensor& x, const Scalar& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::pow_(x_res, y.to<float>());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor put_along_axis_<LazyTensor>(const Tensor& arr, const Tensor& indices, const Tensor& values, int axis, const std::string& reduce) {
  pir::Value arr_res = std::static_pointer_cast<LazyTensor>(arr.impl())->value();
  pir::Value indices_res = std::static_pointer_cast<LazyTensor>(indices.impl())->value();
  pir::Value values_res = std::static_pointer_cast<LazyTensor>(values.impl())->value();
  auto op_res = paddle::dialect::put_along_axis_(arr_res, indices_res, values_res, axis, reduce);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor reciprocal_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::reciprocal_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor relu_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::relu_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor renorm_<LazyTensor>(const Tensor& x, float p, int axis, float max_norm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::renorm_(x_res, p, axis, max_norm);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor round_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::round_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor rsqrt_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::rsqrt_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor scale_<LazyTensor>(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::scale_(x_res, scale.to<float>(), bias, bias_after_scale);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor scale_<LazyTensor>(const Tensor& x, const Tensor& scale_, float bias, bool bias_after_scale) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult scale_res = std::static_pointer_cast<LazyTensor>(scale_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::scale_(x_res, scale_res, bias, bias_after_scale);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor scatter_<LazyTensor>(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value updates_res = std::static_pointer_cast<LazyTensor>(updates.impl())->value();
  auto op_res = paddle::dialect::scatter_(x_res, index_res, updates_res, overwrite);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sigmoid_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sigmoid_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sigmoid_cross_entropy_with_logits_<LazyTensor>(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, bool normalize, int ignore_index) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  paddle::optional<pir::Value> pos_weight_res;
  if(pos_weight) {
    pir::Value pos_weight_res_inner;
    pos_weight_res_inner = std::static_pointer_cast<LazyTensor>(pos_weight.get().impl())->value();
    pos_weight_res = paddle::make_optional<pir::Value>(pos_weight_res_inner);
  }
  auto op_res = paddle::dialect::sigmoid_cross_entropy_with_logits_(x_res, label_res, pos_weight_res, normalize, ignore_index);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sin_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sin_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sinh_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sinh_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor sqrt_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::sqrt_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor squeeze_<LazyTensor>(const Tensor& x, const IntArray& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::squeeze_(x_res, axis.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor squeeze_<LazyTensor>(const Tensor& x, const Tensor& axis_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::squeeze_(x_res, axis_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor tan_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::tan_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor tanh_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::tanh_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor thresholded_relu_<LazyTensor>(const Tensor& x, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::thresholded_relu_(x_res, threshold);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor trunc_<LazyTensor>(const Tensor& input) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  auto op_res = paddle::dialect::trunc_(input_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor uniform_inplace_<LazyTensor>(const Tensor& x, float min, float max, int seed, int diag_num, int diag_step, float diag_val) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::uniform_inplace_(x_res, min, max, seed, diag_num, diag_step, diag_val);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor unsqueeze_<LazyTensor>(const Tensor& x, const IntArray& axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::unsqueeze_(x_res, axis.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor unsqueeze_<LazyTensor>(const Tensor& x, const Tensor& axis_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::unsqueeze_(x_res, axis_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor where_<LazyTensor>(const Tensor& condition, const Tensor& x, const Tensor& y) {
  pir::Value condition_res = std::static_pointer_cast<LazyTensor>(condition.impl())->value();
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::where_(condition_res, x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor add_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::add_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor assign_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::assign_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_allreduce_max_<LazyTensor>(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_allreduce_max_(x_res, ring_id, use_calc_stream, use_model_parallel);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_allreduce_sum_<LazyTensor>(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_allreduce_sum_(x_res, ring_id, use_calc_stream, use_model_parallel);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_broadcast_<LazyTensor>(const Tensor& x, int ring_id, int root, bool use_calc_stream) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_broadcast_(x_res, ring_id, root, use_calc_stream);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_identity_<LazyTensor>(const Tensor& x, int ring_id, bool use_calc_stream, bool use_model_parallel) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_identity_(x_res, ring_id, use_calc_stream, use_model_parallel);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_reduce_sum_<LazyTensor>(const Tensor& x, int ring_id, int root_id, bool use_calc_stream) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_reduce_sum_(x_res, ring_id, root_id, use_calc_stream);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_sync_calc_stream_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_sync_calc_stream_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor c_sync_comm_stream_<LazyTensor>(const Tensor& x) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::c_sync_comm_stream_(x_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor cast_<LazyTensor>(const Tensor& x, DataType dtype) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::cast_(x_res, dtype);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor divide_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::divide_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor equal_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::equal_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor floor_divide_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::floor_divide_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor greater_equal_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::greater_equal_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor greater_than_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::greater_than_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor increment_<LazyTensor>(const Tensor& x, float value) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::increment_(x_res, value);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor less_equal_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::less_equal_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor less_than_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::less_than_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor multiply_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::multiply_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor not_equal_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::not_equal_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor remainder_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::remainder_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor reshape_<LazyTensor>(const Tensor& x, const IntArray& shape) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::reshape_(x_res, shape.GetData());
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor reshape_<LazyTensor>(const Tensor& x, const Tensor& shape_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::OpResult shape_res = std::static_pointer_cast<LazyTensor>(shape_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::reshape_(x_res, shape_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor softmax_<LazyTensor>(const Tensor& x, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::softmax_(x_res, axis);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor subtract_<LazyTensor>(const Tensor& x, const Tensor& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  auto op_res = paddle::dialect::subtract_(x_res, y_res);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor transpose_<LazyTensor>(const Tensor& x, const std::vector<int>& perm) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::transpose_(x_res, perm);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor tril_<LazyTensor>(const Tensor& x, int diagonal) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::tril_(x_res, diagonal);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor triu_<LazyTensor>(const Tensor& x, int diagonal) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  auto op_res = paddle::dialect::triu_(x_res, diagonal);
  Tensor out(std::make_shared<LazyTensor>(op_res));
  return out; 
}

template <>
Tensor acos_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::acos_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor acosh_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::acosh_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor asin_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::asin_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor asinh_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::asinh_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor atan_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::atan_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor atanh_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::atanh_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor bce_loss_grad_<LazyTensor>(const Tensor& input, const Tensor& label, const Tensor& out_grad) {
  pir::Value input_res = std::static_pointer_cast<LazyTensor>(input.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::bce_loss_grad_(input_res, label_res, out_grad_res);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
Tensor ceil_grad_<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::ceil_grad_(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> celu_double_grad_<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::celu_double_grad_(x_res, grad_out_res, grad_x_grad_res, alpha);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor celu_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::celu_grad_(x_res, out_grad_res, alpha);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor clip_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Scalar& min, const Scalar& max) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::clip_grad_(x_res, out_grad_res, min.to<float>(), max.to<float>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor clip_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Tensor& min_, const Tensor& max_) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult min_res = std::static_pointer_cast<LazyTensor>(min_.impl())->value().dyn_cast<pir::OpResult>();
  pir::OpResult max_res = std::static_pointer_cast<LazyTensor>(max_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::clip_grad_(x_res, out_grad_res, min_res, max_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> cos_double_grad_<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> grad_out_res;
  if(grad_out) {
    pir::Value grad_out_res_inner;
    grad_out_res_inner = std::static_pointer_cast<LazyTensor>(grad_out.get().impl())->value();
    grad_out_res = paddle::make_optional<pir::Value>(grad_out_res_inner);
  }
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::cos_double_grad_(x_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor cos_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cos_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> cos_triple_grad_<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> grad_out_forward_res;
  if(grad_out_forward) {
    pir::Value grad_out_forward_res_inner;
    grad_out_forward_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_forward.get().impl())->value();
    grad_out_forward_res = paddle::make_optional<pir::Value>(grad_out_forward_res_inner);
  }
  paddle::optional<pir::Value> grad_x_grad_forward_res;
  if(grad_x_grad_forward) {
    pir::Value grad_x_grad_forward_res_inner;
    grad_x_grad_forward_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad_forward.get().impl())->value();
    grad_x_grad_forward_res = paddle::make_optional<pir::Value>(grad_x_grad_forward_res_inner);
  }
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  paddle::optional<pir::Value> grad_out_grad_grad_res;
  if(grad_out_grad_grad) {
    pir::Value grad_out_grad_grad_res_inner;
    grad_out_grad_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_grad_grad.get().impl())->value();
    grad_out_grad_grad_res = paddle::make_optional<pir::Value>(grad_out_grad_grad_res_inner);
  }
  auto op_res = paddle::dialect::cos_triple_grad_(x_res, grad_out_forward_res, grad_x_grad_forward_res, grad_x_grad_res, grad_out_grad_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_forward_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_x_grad_forward_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, grad_out_forward_grad, grad_x_grad_forward_grad); 
}

template <>
Tensor cosh_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::cosh_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor cross_entropy_with_softmax_grad_<LazyTensor>(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis) {
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value softmax_res = std::static_pointer_cast<LazyTensor>(softmax.impl())->value();
  pir::Value loss_grad_res = std::static_pointer_cast<LazyTensor>(loss_grad.impl())->value();
  auto op_res = paddle::dialect::cross_entropy_with_softmax_grad_(label_res, softmax_res, loss_grad_res, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis);
  Tensor input_grad(std::make_shared<LazyTensor>(op_res));
  return input_grad; 
}

template <>
std::tuple<Tensor, Tensor> elu_double_grad_<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::elu_double_grad_(x_res, grad_out_res, grad_x_grad_res, alpha);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor elu_grad_<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad, float alpha) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::elu_grad_(x_res, out_res, out_grad_res, alpha);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor exp_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::exp_grad_(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor expm1_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::expm1_grad_(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fill_diagonal_tensor_grad_<LazyTensor>(const Tensor& out_grad, int64_t offset, int dim1, int dim2) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fill_diagonal_tensor_grad_(out_grad_res, offset, dim1, dim2);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fill_grad_<LazyTensor>(const Tensor& out_grad, const Scalar& value) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::fill_grad_(out_grad_res, value.to<float>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor fill_grad_<LazyTensor>(const Tensor& out_grad, const Tensor& value_) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult value_res = std::static_pointer_cast<LazyTensor>(value_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::fill_grad_(out_grad_res, value_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor flatten_grad_<LazyTensor>(const Tensor& xshape, const Tensor& out_grad) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::flatten_grad_(xshape_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor floor_grad_<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::floor_grad_(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor gaussian_inplace_grad_<LazyTensor>(const Tensor& out_grad, float mean, float std, int seed) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::gaussian_inplace_grad_(out_grad_res, mean, std, seed);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> group_norm_grad_<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& y, const Tensor& mean, const Tensor& variance, const Tensor& y_grad, float epsilon, int groups, const std::string& data_layout) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> scale_res;
  if(scale) {
    pir::Value scale_res_inner;
    scale_res_inner = std::static_pointer_cast<LazyTensor>(scale.get().impl())->value();
    scale_res = paddle::make_optional<pir::Value>(scale_res_inner);
  }
  paddle::optional<pir::Value> bias_res;
  if(bias) {
    pir::Value bias_res_inner;
    bias_res_inner = std::static_pointer_cast<LazyTensor>(bias.get().impl())->value();
    bias_res = paddle::make_optional<pir::Value>(bias_res_inner);
  }
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value mean_res = std::static_pointer_cast<LazyTensor>(mean.impl())->value();
  pir::Value variance_res = std::static_pointer_cast<LazyTensor>(variance.impl())->value();
  pir::Value y_grad_res = std::static_pointer_cast<LazyTensor>(y_grad.impl())->value();
  auto op_res = paddle::dialect::group_norm_grad_(x_res, scale_res, bias_res, y_res, mean_res, variance_res, y_grad_res, epsilon, groups, data_layout);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor bias_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, scale_grad, bias_grad); 
}

template <>
Tensor hardshrink_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::hardshrink_grad_(x_res, out_grad_res, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor hardsigmoid_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad, float slope, float offset) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::hardsigmoid_grad_(out_res, out_grad_res, slope, offset);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor hardtanh_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, float t_min, float t_max) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::hardtanh_grad_(x_res, out_grad_res, t_min, t_max);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> index_add_grad_<LazyTensor>(const Tensor& index, const Tensor& add_value, const Tensor& out_grad, int axis) {
  pir::Value index_res = std::static_pointer_cast<LazyTensor>(index.impl())->value();
  pir::Value add_value_res = std::static_pointer_cast<LazyTensor>(add_value.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::index_add_grad_(index_res, add_value_res, out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor add_value_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, add_value_grad); 
}

template <>
Tensor leaky_relu_double_grad_<LazyTensor>(const Tensor& x, const Tensor& grad_x_grad, float negative_slope) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::leaky_relu_double_grad_(x_res, grad_x_grad_res, negative_slope);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor leaky_relu_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, float negative_slope) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::leaky_relu_grad_(x_res, out_grad_res, negative_slope);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor log10_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log10_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor log1p_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log1p_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor log2_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log2_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> log_double_grad_<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::log_double_grad_(x_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor log_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::log_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor logsigmoid_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::logsigmoid_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor lu_grad_<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& pivots, const Tensor& out_grad, bool pivot) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value pivots_res = std::static_pointer_cast<LazyTensor>(pivots.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::lu_grad_(x_res, out_res, pivots_res, out_grad_res, pivot);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor margin_cross_entropy_grad_<LazyTensor>(const Tensor& logits, const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale) {
  pir::Value logits_res = std::static_pointer_cast<LazyTensor>(logits.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  pir::Value softmax_res = std::static_pointer_cast<LazyTensor>(softmax.impl())->value();
  pir::Value loss_grad_res = std::static_pointer_cast<LazyTensor>(loss_grad.impl())->value();
  auto op_res = paddle::dialect::margin_cross_entropy_grad_(logits_res, label_res, softmax_res, loss_grad_res, return_softmax, ring_id, rank, nranks, margin1, margin2, margin3, scale);
  Tensor logits_grad(std::make_shared<LazyTensor>(op_res));
  return logits_grad; 
}

template <>
std::tuple<Tensor, Tensor> pow_double_grad_<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const Scalar& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::pow_double_grad_(x_res, grad_out_res, grad_x_grad_res, y.to<float>());
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor pow_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, const Scalar& y) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::pow_grad_(x_res, out_grad_res, y.to<float>());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor reciprocal_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::reciprocal_grad_(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor relu6_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::relu6_grad_(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor relu_double_grad_<LazyTensor>(const Tensor& out, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::relu_double_grad_(out_res, grad_x_grad_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor relu_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::relu_grad_(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor round_grad_<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::round_grad_(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> rsqrt_double_grad_<LazyTensor>(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_x_res = std::static_pointer_cast<LazyTensor>(grad_x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::rsqrt_double_grad_(out_res, grad_x_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_grad, grad_out_grad); 
}

template <>
Tensor rsqrt_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::rsqrt_grad_(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor sigmoid_cross_entropy_with_logits_grad_<LazyTensor>(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, const Tensor& out_grad, bool normalize, int ignore_index) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value label_res = std::static_pointer_cast<LazyTensor>(label.impl())->value();
  paddle::optional<pir::Value> pos_weight_res;
  if(pos_weight) {
    pir::Value pos_weight_res_inner;
    pos_weight_res_inner = std::static_pointer_cast<LazyTensor>(pos_weight.get().impl())->value();
    pos_weight_res = paddle::make_optional<pir::Value>(pos_weight_res_inner);
  }
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sigmoid_cross_entropy_with_logits_grad_(x_res, label_res, pos_weight_res, out_grad_res, normalize, ignore_index);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> sigmoid_double_grad_<LazyTensor>(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value fwd_grad_out_res = std::static_pointer_cast<LazyTensor>(fwd_grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::sigmoid_double_grad_(out_res, fwd_grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor fwd_grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_grad, fwd_grad_out_grad); 
}

template <>
Tensor sigmoid_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sigmoid_grad_(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> sigmoid_triple_grad_<LazyTensor>(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_grad_x, const Tensor& grad_out_grad, const paddle::optional<Tensor>& grad_grad_out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value fwd_grad_out_res = std::static_pointer_cast<LazyTensor>(fwd_grad_out.impl())->value();
  pir::Value grad_grad_x_res = std::static_pointer_cast<LazyTensor>(grad_grad_x.impl())->value();
  pir::Value grad_out_grad_res = std::static_pointer_cast<LazyTensor>(grad_out_grad.impl())->value();
  paddle::optional<pir::Value> grad_grad_out_grad_res;
  if(grad_grad_out_grad) {
    pir::Value grad_grad_out_grad_res_inner;
    grad_grad_out_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_grad_out_grad.get().impl())->value();
    grad_grad_out_grad_res = paddle::make_optional<pir::Value>(grad_grad_out_grad_res_inner);
  }
  auto op_res = paddle::dialect::sigmoid_triple_grad_(out_res, fwd_grad_out_res, grad_grad_x_res, grad_out_grad_res, grad_grad_out_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor fwd_grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_grad_x_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out_grad, fwd_grad_out_grad, grad_grad_x_grad); 
}

template <>
Tensor silu_grad_<LazyTensor>(const Tensor& x, const Tensor& out, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::silu_grad_(x_res, out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> sin_double_grad_<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> grad_out_res;
  if(grad_out) {
    pir::Value grad_out_res_inner;
    grad_out_res_inner = std::static_pointer_cast<LazyTensor>(grad_out.get().impl())->value();
    grad_out_res = paddle::make_optional<pir::Value>(grad_out_res_inner);
  }
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::sin_double_grad_(x_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor sin_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sin_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> sin_triple_grad_<LazyTensor>(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  paddle::optional<pir::Value> grad_out_forward_res;
  if(grad_out_forward) {
    pir::Value grad_out_forward_res_inner;
    grad_out_forward_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_forward.get().impl())->value();
    grad_out_forward_res = paddle::make_optional<pir::Value>(grad_out_forward_res_inner);
  }
  paddle::optional<pir::Value> grad_x_grad_forward_res;
  if(grad_x_grad_forward) {
    pir::Value grad_x_grad_forward_res_inner;
    grad_x_grad_forward_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad_forward.get().impl())->value();
    grad_x_grad_forward_res = paddle::make_optional<pir::Value>(grad_x_grad_forward_res_inner);
  }
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  paddle::optional<pir::Value> grad_out_grad_grad_res;
  if(grad_out_grad_grad) {
    pir::Value grad_out_grad_grad_res_inner;
    grad_out_grad_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_grad_grad.get().impl())->value();
    grad_out_grad_grad_res = paddle::make_optional<pir::Value>(grad_out_grad_grad_res_inner);
  }
  auto op_res = paddle::dialect::sin_triple_grad_(x_res, grad_out_forward_res, grad_x_grad_forward_res, grad_x_grad_res, grad_out_grad_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_forward_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_x_grad_forward_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, grad_out_forward_grad, grad_x_grad_forward_grad); 
}

template <>
Tensor sinh_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sinh_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> softplus_double_grad_<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float beta, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::softplus_double_grad_(x_res, grad_out_res, grad_x_grad_res, beta, threshold);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor softplus_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, float beta, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::softplus_grad_(x_res, out_grad_res, beta, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor softshrink_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::softshrink_grad_(x_res, out_grad_res, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor softsign_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::softsign_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> sqrt_double_grad_<LazyTensor>(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_x_res = std::static_pointer_cast<LazyTensor>(grad_x.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::sqrt_double_grad_(out_res, grad_x_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_grad, grad_out_grad); 
}

template <>
Tensor sqrt_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::sqrt_grad_(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> square_double_grad_<LazyTensor>(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::square_double_grad_(x_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, grad_out_grad); 
}

template <>
Tensor square_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::square_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor squeeze_grad_<LazyTensor>(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::squeeze_grad_(xshape_res, out_grad_res, axis.GetData());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor squeeze_grad_<LazyTensor>(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::squeeze_grad_(xshape_res, out_grad_res, axis_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor tan_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::tan_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor> tanh_double_grad_<LazyTensor>(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::tanh_double_grad_(out_res, grad_out_res, grad_x_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(out_grad, grad_out_grad); 
}

template <>
Tensor tanh_grad_<LazyTensor>(const Tensor& out, const Tensor& out_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::tanh_grad_(out_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor tanh_shrink_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::tanh_shrink_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> tanh_triple_grad_<LazyTensor>(const Tensor& out, const Tensor& grad_out_forward, const Tensor& grad_x_grad_forward, const paddle::optional<Tensor>& grad_out_new_grad, const paddle::optional<Tensor>& grad_out_grad_grad) {
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_out_forward_res = std::static_pointer_cast<LazyTensor>(grad_out_forward.impl())->value();
  pir::Value grad_x_grad_forward_res = std::static_pointer_cast<LazyTensor>(grad_x_grad_forward.impl())->value();
  paddle::optional<pir::Value> grad_out_new_grad_res;
  if(grad_out_new_grad) {
    pir::Value grad_out_new_grad_res_inner;
    grad_out_new_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_new_grad.get().impl())->value();
    grad_out_new_grad_res = paddle::make_optional<pir::Value>(grad_out_new_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_out_grad_grad_res;
  if(grad_out_grad_grad) {
    pir::Value grad_out_grad_grad_res_inner;
    grad_out_grad_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_out_grad_grad.get().impl())->value();
    grad_out_grad_grad_res = paddle::make_optional<pir::Value>(grad_out_grad_grad_res_inner);
  }
  auto op_res = paddle::dialect::tanh_triple_grad_(out_res, grad_out_forward_res, grad_x_grad_forward_res, grad_out_new_grad_res, grad_out_grad_grad_res);
  auto op_res_0 = std::get<0>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_out_forward_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_x_grad_forward_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(out_grad, grad_out_forward_grad, grad_x_grad_forward_grad); 
}

template <>
Tensor thresholded_relu_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::thresholded_relu_grad_(x_res, out_grad_res, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor uniform_inplace_grad_<LazyTensor>(const Tensor& out_grad, float min, float max, int seed, int diag_num, int diag_step, float diag_val) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::uniform_inplace_grad_(out_grad_res, min, max, seed, diag_num, diag_step, diag_val);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor unsqueeze_grad_<LazyTensor>(const Tensor& xshape, const Tensor& out_grad, const IntArray& axis) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::unsqueeze_grad_(xshape_res, out_grad_res, axis.GetData());
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor unsqueeze_grad_<LazyTensor>(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  pir::OpResult axis_res = std::static_pointer_cast<LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>();
  auto op_res = paddle::dialect::unsqueeze_grad_(xshape_res, out_grad_res, axis_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor add_double_grad_<LazyTensor>(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis) {
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  auto op_res = paddle::dialect::add_double_grad_(y_res, grad_out_res, grad_x_grad_res, grad_y_grad_res, axis);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
std::tuple<Tensor, Tensor> add_grad_<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::add_grad_(x_res, y_res, out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
std::tuple<Tensor, Tensor> add_triple_grad_<LazyTensor>(const Tensor& grad_grad_x, const Tensor& grad_grad_y, const Tensor& grad_grad_out_grad, int axis) {
  pir::Value grad_grad_x_res = std::static_pointer_cast<LazyTensor>(grad_grad_x.impl())->value();
  pir::Value grad_grad_y_res = std::static_pointer_cast<LazyTensor>(grad_grad_y.impl())->value();
  pir::Value grad_grad_out_grad_res = std::static_pointer_cast<LazyTensor>(grad_grad_out_grad.impl())->value();
  auto op_res = paddle::dialect::add_triple_grad_(grad_grad_x_res, grad_grad_y_res, grad_grad_out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor grad_grad_x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor grad_grad_y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(grad_grad_x_grad, grad_grad_y_grad); 
}

template <>
Tensor assign_out__grad_<LazyTensor>(const Tensor& out_grad) {
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::assign_out__grad_(out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> batch_norm_double_grad_<LazyTensor>(const Tensor& x, const Tensor& scale, const paddle::optional<Tensor>& out_mean, const paddle::optional<Tensor>& out_variance, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value scale_res = std::static_pointer_cast<LazyTensor>(scale.impl())->value();
  paddle::optional<pir::Value> out_mean_res;
  if(out_mean) {
    pir::Value out_mean_res_inner;
    out_mean_res_inner = std::static_pointer_cast<LazyTensor>(out_mean.get().impl())->value();
    out_mean_res = paddle::make_optional<pir::Value>(out_mean_res_inner);
  }
  paddle::optional<pir::Value> out_variance_res;
  if(out_variance) {
    pir::Value out_variance_res_inner;
    out_variance_res_inner = std::static_pointer_cast<LazyTensor>(out_variance.get().impl())->value();
    out_variance_res = paddle::make_optional<pir::Value>(out_variance_res_inner);
  }
  pir::Value saved_mean_res = std::static_pointer_cast<LazyTensor>(saved_mean.impl())->value();
  pir::Value saved_variance_res = std::static_pointer_cast<LazyTensor>(saved_variance.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_scale_grad_res;
  if(grad_scale_grad) {
    pir::Value grad_scale_grad_res_inner;
    grad_scale_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_scale_grad.get().impl())->value();
    grad_scale_grad_res = paddle::make_optional<pir::Value>(grad_scale_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_bias_grad_res;
  if(grad_bias_grad) {
    pir::Value grad_bias_grad_res_inner;
    grad_bias_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_bias_grad.get().impl())->value();
    grad_bias_grad_res = paddle::make_optional<pir::Value>(grad_bias_grad_res_inner);
  }
  auto op_res = paddle::dialect::batch_norm_double_grad_(x_res, scale_res, out_mean_res, out_variance_res, saved_mean_res, saved_variance_res, grad_out_res, grad_x_grad_res, grad_scale_grad_res, grad_bias_grad_res, momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor scale_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, scale_grad, grad_out_grad); 
}

template <>
std::tuple<Tensor, Tensor, Tensor> divide_double_grad_<LazyTensor>(const Tensor& y, const Tensor& out, const Tensor& grad_x, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis) {
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_res = std::static_pointer_cast<LazyTensor>(out.impl())->value();
  pir::Value grad_x_res = std::static_pointer_cast<LazyTensor>(grad_x.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  auto op_res = paddle::dialect::divide_double_grad_(y_res, out_res, grad_x_res, grad_x_grad_res, grad_y_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor out_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(y_grad, out_grad, grad_out_grad); 
}

template <>
Tensor hardswish_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::hardswish_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor mish_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad, float threshold) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::mish_grad_(x_res, out_grad_res, threshold);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
std::tuple<Tensor, Tensor, Tensor> multiply_double_grad_<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  auto op_res = paddle::dialect::multiply_double_grad_(x_res, y_res, grad_out_res, grad_x_grad_res, grad_y_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  auto op_res_2 = std::get<2>(op_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res_2));
  return std::make_tuple(x_grad, y_grad, grad_out_grad); 
}

template <>
Tensor reshape_double_grad_<LazyTensor>(const Tensor& grad_out, const Tensor& grad_x_grad) {
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  pir::Value grad_x_grad_res = std::static_pointer_cast<LazyTensor>(grad_x_grad.impl())->value();
  auto op_res = paddle::dialect::reshape_double_grad_(grad_out_res, grad_x_grad_res);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
Tensor reshape_grad_<LazyTensor>(const Tensor& xshape, const Tensor& out_grad) {
  pir::Value xshape_res = std::static_pointer_cast<LazyTensor>(xshape.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::reshape_grad_(xshape_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}

template <>
Tensor subtract_double_grad_<LazyTensor>(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis) {
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value grad_out_res = std::static_pointer_cast<LazyTensor>(grad_out.impl())->value();
  paddle::optional<pir::Value> grad_x_grad_res;
  if(grad_x_grad) {
    pir::Value grad_x_grad_res_inner;
    grad_x_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_x_grad.get().impl())->value();
    grad_x_grad_res = paddle::make_optional<pir::Value>(grad_x_grad_res_inner);
  }
  paddle::optional<pir::Value> grad_y_grad_res;
  if(grad_y_grad) {
    pir::Value grad_y_grad_res_inner;
    grad_y_grad_res_inner = std::static_pointer_cast<LazyTensor>(grad_y_grad.get().impl())->value();
    grad_y_grad_res = paddle::make_optional<pir::Value>(grad_y_grad_res_inner);
  }
  auto op_res = paddle::dialect::subtract_double_grad_(y_res, grad_out_res, grad_x_grad_res, grad_y_grad_res, axis);
  Tensor grad_out_grad(std::make_shared<LazyTensor>(op_res));
  return grad_out_grad; 
}

template <>
std::tuple<Tensor, Tensor> subtract_grad_<LazyTensor>(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value y_res = std::static_pointer_cast<LazyTensor>(y.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::subtract_grad_(x_res, y_res, out_grad_res, axis);
  auto op_res_0 = std::get<0>(op_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res_0));
  auto op_res_1 = std::get<1>(op_res);
  Tensor y_grad(std::make_shared<LazyTensor>(op_res_1));
  return std::make_tuple(x_grad, y_grad); 
}

template <>
Tensor swish_grad_<LazyTensor>(const Tensor& x, const Tensor& out_grad) {
  pir::Value x_res = std::static_pointer_cast<LazyTensor>(x.impl())->value();
  pir::Value out_grad_res = std::static_pointer_cast<LazyTensor>(out_grad.impl())->value();
  auto op_res = paddle::dialect::swish_grad_(x_res, out_grad_res);
  Tensor x_grad(std::make_shared<LazyTensor>(op_res));
  return x_grad; 
}


}  // namespace backend
}  // namespace primitive
}  // namespace paddle
