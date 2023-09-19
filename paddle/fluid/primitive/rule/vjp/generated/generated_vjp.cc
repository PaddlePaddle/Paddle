// Auto Generated, DO NOT EDIT!

#include "paddle/fluid/primitive/rule/vjp/generated/generated_vjp.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/fluid/primitive/backend/backend.h"
#include "paddle/fluid/primitive/rule/vjp/details.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"
#include "paddle/pir/core/operation.h"
#include "paddle/phi/core/flags.h"
#include "paddle/utils/optional.h"

PHI_DECLARE_string(tensor_operants_mode);

namespace paddle {
namespace primitive {


std::vector<std::vector<paddle::Tensor>> concat_vjp(const std::vector<Tensor>& x, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    std::vector<paddle::Tensor*> x_grad(stop_gradients[0].size(), nullptr);
    for (size_t i=0; i< stop_gradients[0].size(); i++ ) {
      x_grad[i] =  !stop_gradients[0][i] ?  &vjp_res[0][i] : nullptr;
    }
    auto* axis_define_op = std::static_pointer_cast<primitive::LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>().owner();
    if(axis_define_op->name() != "pd_op.full") {
      PADDLE_THROW(platform::errors::Unimplemented(
          "We don't support dynamic tensors attribute axis for concat_grad composite "
          "for now. "));
    }
    auto axis = axis_define_op->attribute("value").dyn_cast<paddle::dialect::ScalarAttribute>().data();

    details::concat_grad<LazyTensor>(x, out_grad, axis, x_grad);
  } else {
    auto op_res = backend::concat_grad<LazyTensor>(x, out_grad, axis_);
    vjp_res[0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> erf_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::erf_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> exp_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::exp_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> expand_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& shape_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::expand_grad<LazyTensor>(x, out_grad, shape_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> gelu_vjp(const Tensor& x, const Tensor& out_grad, bool approximate, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::gelu_grad<LazyTensor>(x, out_grad, approximate, x_grad);
  } else {
    auto op_res = backend::gelu_grad<LazyTensor>(x, out_grad, approximate);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> layer_norm_vjp(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& mean, const Tensor& variance, const Tensor& out_grad, float epsilon, int begin_norm_axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    paddle::Tensor* scale_grad = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr; 
    paddle::Tensor* bias_grad = !stop_gradients[2][0] ? &vjp_res[2][0] : nullptr; 

    details::layer_norm_grad<LazyTensor>(x, scale, bias, mean, variance, out_grad, epsilon, begin_norm_axis, x_grad, scale_grad, bias_grad);
  } else {
    auto op_res = backend::layer_norm_grad<LazyTensor>(x, scale, bias, mean, variance, out_grad, epsilon, begin_norm_axis);
    vjp_res[0][0] = std::get<0>(op_res);
    vjp_res[1][0] = std::get<1>(op_res);
    vjp_res[2][0] = std::get<2>(op_res);
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pow_vjp(const Tensor& x, const Tensor& out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pow_grad<LazyTensor>(x, out_grad, y);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> rsqrt_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::rsqrt_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> scale_vjp(const Tensor& out_grad, const Tensor& scale_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::scale<LazyTensor>(out_grad, scale_, 0.0f, true);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> silu_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::silu_grad<LazyTensor>(x, out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> square_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::square_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tanh_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::tanh_grad<LazyTensor>(out, out_grad, x_grad);
  } else {
    auto op_res = backend::tanh_grad<LazyTensor>(out, out_grad);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> add_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    paddle::Tensor* y_grad = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr; 

    details::add_grad<LazyTensor>(x, y, out_grad, axis, x_grad, y_grad);
  } else {
    auto op_res = backend::add_grad<LazyTensor>(x, y, out_grad, axis);
    vjp_res[0][0] = std::get<0>(op_res);
    vjp_res[1][0] = std::get<1>(op_res);
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cast_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::cast_grad<LazyTensor>(x, out_grad, x_grad);
  } else {
    auto op_res = backend::cast<LazyTensor>(out_grad, x.dtype());
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> divide_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    paddle::Tensor* y_grad = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr; 

    details::divide_grad<LazyTensor>(x, y, out, out_grad, axis, x_grad, y_grad);
  } else {
    auto op_res = backend::divide_grad<LazyTensor>(x, y, out, out_grad, axis);
    vjp_res[0][0] = std::get<0>(op_res);
    vjp_res[1][0] = std::get<1>(op_res);
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> dropout_vjp(const Tensor& mask, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::dropout_grad<LazyTensor>(mask, out_grad, p, is_test, mode);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> elementwise_pow_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    paddle::Tensor* y_grad = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr; 

    details::elementwise_pow_grad<LazyTensor>(x, y, out_grad, x_grad, y_grad);
  } else {
    auto op_res = backend::elementwise_pow_grad<LazyTensor>(x, y, out_grad);
    vjp_res[0][0] = std::get<0>(op_res);
    vjp_res[1][0] = std::get<1>(op_res);
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> embedding_vjp(const Tensor& x, const Tensor& weight, const Tensor& out_grad, int64_t padding_idx, bool sparse, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::embedding_grad<LazyTensor>(x, weight, out_grad, padding_idx, sparse);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fused_softmax_mask_upper_triangle_vjp(const Tensor& Out, const Tensor& Out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fused_softmax_mask_upper_triangle_grad<LazyTensor>(Out, Out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> matmul_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x, bool transpose_y, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::matmul_grad<LazyTensor>(x, y, out_grad, transpose_x, transpose_y);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> mean_vjp(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::mean_grad<LazyTensor>(x, out_grad, axis, keepdim, reduce_all);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> multiply_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    paddle::Tensor* y_grad = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr; 

    details::multiply_grad<LazyTensor>(x, y, out_grad, axis, x_grad, y_grad);
  } else {
    auto op_res = backend::multiply_grad<LazyTensor>(x, y, out_grad, axis);
    vjp_res[0][0] = std::get<0>(op_res);
    vjp_res[1][0] = std::get<1>(op_res);
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> reshape_vjp(const Tensor& xshape, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::reshape_grad<LazyTensor>(xshape, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> slice_vjp(const Tensor& input, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::slice_grad<LazyTensor>(input, out_grad, starts_, ends_, axes, infer_flags, decrease_axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> softmax_vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::softmax_grad<LazyTensor>(out, out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> split_vjp(const std::vector<Tensor>& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    auto* axis_define_op = std::static_pointer_cast<primitive::LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>().owner();
    if(axis_define_op->name() != "pd_op.full") {
      PADDLE_THROW(platform::errors::Unimplemented(
          "We don't support dynamic tensors attribute axis for split_grad composite "
          "for now. "));
    }
    auto axis = axis_define_op->attribute("value").dyn_cast<paddle::dialect::ScalarAttribute>().data();

    details::split_grad<LazyTensor>(out_grad, axis, x_grad);
  } else {
    auto op_res = backend::concat<LazyTensor>(out_grad, axis_);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> split_with_num_vjp(const std::vector<Tensor>& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::concat<LazyTensor>(out_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> subtract_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::subtract_grad<LazyTensor>(x, y, out_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sum_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    auto* axis_define_op = std::static_pointer_cast<primitive::LazyTensor>(axis_.impl())->value().dyn_cast<pir::OpResult>().owner();
    if(axis_define_op->name() != "pd_op.full_int_array"){
      PADDLE_THROW(platform::errors::Unimplemented(
          "We don't support dynamic tensors attribute axis for sum_grad composite "
          "for now. "));
    }
    auto axis = axis_define_op->attribute("value").dyn_cast<paddle::dialect::IntArrayAttribute>().data();

    details::sum_grad<LazyTensor>(x, out_grad, axis, keepdim, reduce_all, x_grad);
  } else {
    auto op_res = backend::sum_grad<LazyTensor>(x, out_grad, axis_, keepdim, reduce_all);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> transpose_vjp(const Tensor& out_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::transpose_grad<LazyTensor>(out_grad, perm, x_grad);
  } else {
    auto op_res = backend::transpose_grad<LazyTensor>(out_grad, perm);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> slice_grad_vjp(const Tensor& grad_input_grad, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::slice<LazyTensor>(grad_input_grad, starts_, ends_, axes, infer_flags, decrease_axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> erf__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::erf_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> exp__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::exp_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pow__vjp(const Tensor& x, const Tensor& out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pow_grad<LazyTensor>(x, out_grad, y);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> rsqrt__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::rsqrt_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> scale__vjp(const Tensor& out_grad, const Tensor& scale_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::scale<LazyTensor>(out_grad, scale_, 0.0f, true);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tanh__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::tanh_grad<LazyTensor>(out, out_grad, x_grad);
  } else {
    auto op_res = backend::tanh_grad<LazyTensor>(out, out_grad);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> add__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    paddle::Tensor* y_grad = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr; 

    details::add_grad<LazyTensor>(x, y, out_grad, axis, x_grad, y_grad);
  } else {
    auto op_res = backend::add_grad<LazyTensor>(x, y, out_grad, axis);
    vjp_res[0][0] = std::get<0>(op_res);
    vjp_res[1][0] = std::get<1>(op_res);
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cast__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::cast_grad<LazyTensor>(x, out_grad, x_grad);
  } else {
    auto op_res = backend::cast<LazyTensor>(out_grad, x.dtype());
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> divide__vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    paddle::Tensor* y_grad = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr; 

    details::divide_grad<LazyTensor>(x, y, out, out_grad, axis, x_grad, y_grad);
  } else {
    auto op_res = backend::divide_grad<LazyTensor>(x, y, out, out_grad, axis);
    vjp_res[0][0] = std::get<0>(op_res);
    vjp_res[1][0] = std::get<1>(op_res);
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> multiply__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    FLAGS_tensor_operants_mode = "static";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    paddle::Tensor* y_grad = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr; 

    details::multiply_grad<LazyTensor>(x, y, out_grad, axis, x_grad, y_grad);
  } else {
    auto op_res = backend::multiply_grad<LazyTensor>(x, y, out_grad, axis);
    vjp_res[0][0] = std::get<0>(op_res);
    vjp_res[1][0] = std::get<1>(op_res);
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> reshape__vjp(const Tensor& xshape, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::reshape_grad<LazyTensor>(xshape, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> softmax__vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::softmax_grad<LazyTensor>(out, out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> subtract__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::subtract_grad<LazyTensor>(x, y, out_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}



}  // namespace primitive
}  // namespace paddle
