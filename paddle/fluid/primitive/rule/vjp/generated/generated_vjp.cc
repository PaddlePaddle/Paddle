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


std::vector<std::vector<paddle::Tensor>> abs_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::abs_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> acos_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::acos_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> acosh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::acosh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> addmm_vjp(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha, float beta, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::addmm_grad<LazyTensor>(input, x, y, out_grad, alpha, beta);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> affine_grid_vjp(const Tensor& input, const Tensor& output_grad, const Tensor& output_shape_, bool align_corners, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::affine_grid_grad<LazyTensor>(input, output_grad, output_shape_, align_corners);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> angle_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::angle_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> argsort_vjp(const Tensor& indices, const Tensor& x, const Tensor& out_grad, int axis, bool descending, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::argsort_grad<LazyTensor>(indices, x, out_grad, axis, descending);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> as_complex_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::as_real<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> as_real_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::as_complex<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> as_strided_vjp(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims, const std::vector<int64_t>& stride, int64_t offset, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::as_strided_grad<LazyTensor>(input, out_grad, dims, stride, offset);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> asin_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::asin_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> asinh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::asinh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> atan_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::atan_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> atan2_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::atan2_grad<LazyTensor>(x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> atanh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::atanh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> bce_loss_vjp(const Tensor& input, const Tensor& label, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::bce_loss_grad<LazyTensor>(input, label, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> bicubic_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::bicubic_interp_grad<LazyTensor>(x, out_size, size_tensor, scale_tensor, output_grad, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> bilinear_vjp(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::bilinear_grad<LazyTensor>(x, y, weight, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res[3][0] = std::get<3>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> bilinear_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::bilinear_interp_grad<LazyTensor>(x, out_size, size_tensor, scale_tensor, output_grad, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> bmm_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::bmm_grad<LazyTensor>(x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> broadcast_tensors_vjp(const std::vector<Tensor>& input, const std::vector<Tensor>& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::broadcast_tensors_grad<LazyTensor>(input, out_grad);
  vjp_res[0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> ceil_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::ceil_grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> celu_vjp(const Tensor& x, const Tensor& out_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::celu_grad<LazyTensor>(x, out_grad, alpha);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cholesky_vjp(const Tensor& out, const Tensor& out_grad, bool upper, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cholesky_grad<LazyTensor>(out, out_grad, upper);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cholesky_solve_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cholesky_solve_grad<LazyTensor>(x, y, out, out_grad, upper);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> clip_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& min_, const Tensor& max_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::clip_grad<LazyTensor>(x, out_grad, min_, max_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> complex_vjp(const Tensor& real, const Tensor& imag, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::complex_grad<LazyTensor>(real, imag, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> concat_vjp(const std::vector<Tensor>& x, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "concat_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op concat_grad";
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

std::vector<std::vector<paddle::Tensor>> conj_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::conj<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> conv2d_vjp(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::conv2d_grad<LazyTensor>(input, filter, out_grad, strides, paddings, padding_algorithm, dilations, groups, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> conv3d_vjp(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::conv3d_grad<LazyTensor>(input, filter, out_grad, strides, paddings, padding_algorithm, groups, dilations, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> conv3d_transpose_vjp(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::conv3d_transpose_grad<LazyTensor>(x, filter, out_grad, strides, paddings, output_padding, output_size, padding_algorithm, groups, dilations, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cos_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cos_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cosh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cosh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> crop_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& offsets_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::crop_grad<LazyTensor>(x, out_grad, offsets_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cross_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cross_grad<LazyTensor>(x, y, out_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cross_entropy_with_softmax_vjp(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cross_entropy_with_softmax_grad<LazyTensor>(label, softmax, loss_grad, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cummax_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, int dtype, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cummax_grad<LazyTensor>(x, indices, out_grad, axis, dtype);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cummin_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, int dtype, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cummin_grad<LazyTensor>(x, indices, out_grad, axis, dtype);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cumprod_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int dim, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cumprod_grad<LazyTensor>(x, out, out_grad, dim);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cumsum_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool flatten, bool exclusive, bool reverse, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cumsum_grad<LazyTensor>(x, out_grad, axis_, flatten, exclusive, reverse);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> depthwise_conv2d_vjp(const Tensor& input, const Tensor& filter, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::depthwise_conv2d_grad<LazyTensor>(input, filter, out_grad, strides, paddings, padding_algorithm, groups, dilations, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> det_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::det_grad<LazyTensor>(x, out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> diag_vjp(const Tensor& x, const Tensor& out_grad, int offset, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::diag_grad<LazyTensor>(x, out_grad, offset);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> diagonal_vjp(const Tensor& x, const Tensor& out_grad, int offset, int axis1, int axis2, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::diagonal_grad<LazyTensor>(x, out_grad, offset, axis1, axis2);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> digamma_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::digamma_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> dist_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, float p, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::dist_grad<LazyTensor>(x, y, out, out_grad, p);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> dot_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::dot_grad<LazyTensor>(x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> eig_vjp(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::eig_grad<LazyTensor>(out_w, out_v, out_w_grad, out_v_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> eigh_vjp(const Tensor& out_w, const Tensor& out_v, const Tensor& out_w_grad, const Tensor& out_v_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::eigh_grad<LazyTensor>(out_w, out_v, out_w_grad, out_v_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> eigvalsh_vjp(const Tensor& eigenvectors, const Tensor& eigenvalues_grad, const std::string& uplo, bool is_test, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::eigvalsh_grad<LazyTensor>(eigenvectors, eigenvalues_grad, uplo, is_test);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> elu_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::elu_grad<LazyTensor>(x, out, out_grad, alpha);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
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

std::vector<std::vector<paddle::Tensor>> erfinv_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::erfinv_grad<LazyTensor>(out, out_grad);
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

std::vector<std::vector<paddle::Tensor>> expand_as_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& target_shape, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::expand_as_grad<LazyTensor>(x, out_grad, target_shape);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> expm1_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::expm1_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fft_c2c_vjp(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fft_c2c_grad<LazyTensor>(out_grad, axes, normalization, forward);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fft_c2r_vjp(const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, int64_t last_dim_size, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fft_c2r_grad<LazyTensor>(out_grad, axes, normalization, forward, last_dim_size);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fft_r2c_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int64_t>& axes, const std::string& normalization, bool forward, bool onesided, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fft_r2c_grad<LazyTensor>(x, out_grad, axes, normalization, forward, onesided);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fill_vjp(const Tensor& out_grad, const Tensor& value_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fill_grad<LazyTensor>(out_grad, value_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fill_diagonal_vjp(const Tensor& out_grad, float value, int offset, bool wrap, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fill_diagonal_grad<LazyTensor>(out_grad, value, offset, wrap);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fill_diagonal_tensor_vjp(const Tensor& out_grad, int64_t offset, int dim1, int dim2, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fill_diagonal_tensor_grad<LazyTensor>(out_grad, offset, dim1, dim2);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> flash_attn_vjp(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, float dropout, bool causal, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::flash_attn_grad<LazyTensor>(q, k, v, out, softmax_lse, seed_offset, attn_mask, out_grad, dropout, causal);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> flash_attn_unpadded_vjp(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k, const Tensor& out, const Tensor& softmax_lse, const Tensor& seed_offset, const paddle::optional<Tensor>& attn_mask, const Tensor& out_grad, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout, bool causal, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::flash_attn_unpadded_grad<LazyTensor>(q, k, v, cu_seqlens_q, cu_seqlens_k, out, softmax_lse, seed_offset, attn_mask, out_grad, max_seqlen_q, max_seqlen_k, scale, dropout, causal);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> flatten_vjp(const Tensor& xshape, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::flatten_grad<LazyTensor>(xshape, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> flip_vjp(const Tensor& out_grad, const std::vector<int>& axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::flip<LazyTensor>(out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> floor_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::floor_grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fmax_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fmax_grad<LazyTensor>(x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fmin_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fmin_grad<LazyTensor>(x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fold_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& output_sizes, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fold_grad<LazyTensor>(x, out_grad, output_sizes, kernel_sizes, strides, paddings, dilations);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> frame_vjp(const Tensor& x, const Tensor& out_grad, int frame_length, int hop_length, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::frame_grad<LazyTensor>(x, out_grad, frame_length, hop_length, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> gather_vjp(const Tensor& x, const Tensor& index, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::gather_grad<LazyTensor>(x, index, out_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> gather_nd_vjp(const Tensor& x, const Tensor& index, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::gather_nd_grad<LazyTensor>(x, index, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> gaussian_inplace_vjp(const Tensor& out_grad, float mean, float std, int seed, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::gaussian_inplace_grad<LazyTensor>(out_grad, mean, std, seed);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> gelu_vjp(const Tensor& x, const Tensor& out_grad, bool approximate, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "gelu_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op gelu_grad";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::gelu_grad<LazyTensor>(x, out_grad, approximate, x_grad);
  } else {
    auto op_res = backend::gelu_grad<LazyTensor>(x, out_grad, approximate);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> grid_sample_vjp(const Tensor& x, const Tensor& grid, const Tensor& out_grad, const std::string& mode, const std::string& padding_mode, bool align_corners, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::grid_sample_grad<LazyTensor>(x, grid, out_grad, mode, padding_mode, align_corners);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> group_norm_vjp(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& y, const Tensor& mean, const Tensor& variance, const Tensor& y_grad, float epsilon, int groups, const std::string& data_layout, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::group_norm_grad<LazyTensor>(x, scale, bias, y, mean, variance, y_grad, epsilon, groups, data_layout);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> gumbel_softmax_vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::gumbel_softmax_grad<LazyTensor>(out, out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> hardshrink_vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::hardshrink_grad<LazyTensor>(x, out_grad, threshold);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> hardsigmoid_vjp(const Tensor& out, const Tensor& out_grad, float slope, float offset, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::hardsigmoid_grad<LazyTensor>(out, out_grad, slope, offset);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> hardtanh_vjp(const Tensor& x, const Tensor& out_grad, float t_min, float t_max, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::hardtanh_grad<LazyTensor>(x, out_grad, t_min, t_max);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> heaviside_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::heaviside_grad<LazyTensor>(x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> huber_loss_vjp(const Tensor& residual, const Tensor& out_grad, float delta, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::huber_loss_grad<LazyTensor>(residual, out_grad, delta);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> i0_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::i0_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> i0e_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::i0e_grad<LazyTensor>(x, out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> i1_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::i1_grad<LazyTensor>(x, out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> i1e_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::i1e_grad<LazyTensor>(x, out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> imag_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::imag_grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> index_add_vjp(const Tensor& index, const Tensor& add_value, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::index_add_grad<LazyTensor>(index, add_value, out_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> index_put_vjp(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, const Tensor& out_grad, bool accumulate, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::index_put_grad<LazyTensor>(x, indices, value, out_grad, accumulate);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> index_sample_vjp(const Tensor& x, const Tensor& index, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::index_sample_grad<LazyTensor>(x, index, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> index_select_vjp(const Tensor& x, const Tensor& index, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::index_select_grad<LazyTensor>(x, index, out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> index_select_strided_vjp(const Tensor& x, const Tensor& out_grad, int64_t index, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::index_select_strided_grad<LazyTensor>(x, out_grad, index, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> instance_norm_vjp(const Tensor& x, const paddle::optional<Tensor>& scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& y_grad, float epsilon, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::instance_norm_grad<LazyTensor>(x, scale, saved_mean, saved_variance, y_grad, epsilon);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> inverse_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::inverse_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> kldiv_loss_vjp(const Tensor& x, const Tensor& label, const Tensor& out_grad, const std::string& reduction, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::kldiv_loss_grad<LazyTensor>(x, label, out_grad, reduction);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> kron_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::kron_grad<LazyTensor>(x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> kthvalue_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int k, int axis, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::kthvalue_grad<LazyTensor>(x, indices, out_grad, k, axis, keepdim);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> label_smooth_vjp(const Tensor& out_grad, float epsilon, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::label_smooth_grad<LazyTensor>(out_grad, epsilon);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> layer_norm_vjp(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& mean, const Tensor& variance, const Tensor& out_grad, float epsilon, int begin_norm_axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "layer_norm_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op layer_norm_grad";
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

std::vector<std::vector<paddle::Tensor>> leaky_relu_vjp(const Tensor& x, const Tensor& out_grad, float negative_slope, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::leaky_relu_grad<LazyTensor>(x, out_grad, negative_slope);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> lerp_vjp(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::lerp_grad<LazyTensor>(x, y, weight, out, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> lgamma_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::lgamma_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> linear_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::linear_interp_grad<LazyTensor>(x, out_size, size_tensor, scale_tensor, output_grad, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log10_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log10_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log1p_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log1p_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log2_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log2_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log_loss_vjp(const Tensor& input, const Tensor& label, const Tensor& out_grad, float epsilon, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log_loss_grad<LazyTensor>(input, label, out_grad, epsilon);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log_softmax_vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log_softmax_grad<LazyTensor>(out, out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> logcumsumexp_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int axis, bool flatten, bool exclusive, bool reverse, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::logcumsumexp_grad<LazyTensor>(x, out, out_grad, axis, flatten, exclusive, reverse);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> logit_vjp(const Tensor& x, const Tensor& out_grad, float eps, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::logit_grad<LazyTensor>(x, out_grad, eps);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> logsigmoid_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::logsigmoid_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> lu_vjp(const Tensor& x, const Tensor& out, const Tensor& pivots, const Tensor& out_grad, bool pivot, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::lu_grad<LazyTensor>(x, out, pivots, out_grad, pivot);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> lu_unpack_vjp(const Tensor& x, const Tensor& y, const Tensor& l, const Tensor& u, const Tensor& pmat, const Tensor& l_grad, const Tensor& u_grad, bool unpack_ludata, bool unpack_pivots, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::lu_unpack_grad<LazyTensor>(x, y, l, u, pmat, l_grad, u_grad, unpack_ludata, unpack_pivots);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> margin_cross_entropy_vjp(const Tensor& logits, const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::margin_cross_entropy_grad<LazyTensor>(logits, label, softmax, loss_grad, return_softmax, ring_id, rank, nranks, margin1, margin2, margin3, scale);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> masked_select_vjp(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::masked_select_grad<LazyTensor>(x, mask, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> matrix_power_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int n, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::matrix_power_grad<LazyTensor>(x, out, out_grad, n);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> max_pool2d_with_index_vjp(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::max_pool2d_with_index_grad<LazyTensor>(x, mask, out_grad, kernel_size, strides, paddings, global_pooling, adaptive);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> max_pool3d_with_index_vjp(const Tensor& x, const Tensor& mask, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::max_pool3d_with_index_grad<LazyTensor>(x, mask, out_grad, kernel_size, strides, paddings, global_pooling, adaptive);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> maxout_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int groups, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::maxout_grad<LazyTensor>(x, out, out_grad, groups, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> mean_all_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::mean_all_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> memory_efficient_attention_vjp(const Tensor& query, const Tensor& key, const Tensor& value, const paddle::optional<Tensor>& bias, const paddle::optional<Tensor>& cu_seqlens_q, const paddle::optional<Tensor>& cu_seqlens_k, const Tensor& output, const Tensor& logsumexp, const Tensor& seed_and_offset, const Tensor& output_grad, const Scalar& max_seqlen_q, const Scalar& max_seqlen_k, bool causal, double dropout_p, float scale, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::memory_efficient_attention_grad<LazyTensor>(query, key, value, bias, cu_seqlens_q, cu_seqlens_k, output, logsumexp, seed_and_offset, output_grad, max_seqlen_q, max_seqlen_k, causal, dropout_p, scale);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res[3][0] = std::get<3>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> meshgrid_vjp(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::meshgrid_grad<LazyTensor>(inputs, outputs_grad);
  vjp_res[0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> mode_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, int axis, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::mode_grad<LazyTensor>(x, indices, out_grad, axis, keepdim);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> multi_dot_vjp(const std::vector<Tensor>& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::multi_dot_grad<LazyTensor>(x, out_grad);
  vjp_res[0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> multiplex_vjp(const std::vector<Tensor>& inputs, const Tensor& index, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::multiplex_grad<LazyTensor>(inputs, index, out_grad);
  vjp_res[0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> mv_vjp(const Tensor& x, const Tensor& vec, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::mv_grad<LazyTensor>(x, vec, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> nanmedian_vjp(const Tensor& x, const Tensor& medians, const Tensor& out_grad, const IntArray& axis, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::nanmedian_grad<LazyTensor>(x, medians, out_grad, axis, keepdim);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> nearest_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::nearest_interp_grad<LazyTensor>(x, out_size, size_tensor, scale_tensor, output_grad, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> nll_loss_vjp(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, const Tensor& total_weight, const Tensor& out_grad, int64_t ignore_index, const std::string& reduction, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::nll_loss_grad<LazyTensor>(input, label, weight, total_weight, out_grad, ignore_index, reduction);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> overlap_add_vjp(const Tensor& x, const Tensor& out_grad, int hop_length, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::overlap_add_grad<LazyTensor>(x, out_grad, hop_length, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> p_norm_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, float porder, int axis, float epsilon, bool keepdim, bool asvector, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::p_norm_grad<LazyTensor>(x, out, out_grad, porder, axis, epsilon, keepdim, asvector);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pad3d_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& paddings_, const std::string& mode, float pad_value, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pad3d_grad<LazyTensor>(x, out_grad, paddings_, mode, pad_value, data_format);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pixel_shuffle_vjp(const Tensor& out_grad, int upscale_factor, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pixel_shuffle_grad<LazyTensor>(out_grad, upscale_factor, data_format);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pixel_unshuffle_vjp(const Tensor& out_grad, int downscale_factor, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pixel_unshuffle_grad<LazyTensor>(out_grad, downscale_factor, data_format);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> poisson_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::poisson_grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> polygamma_vjp(const Tensor& x, const Tensor& out_grad, int n, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::polygamma_grad<LazyTensor>(x, out_grad, n);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
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

std::vector<std::vector<paddle::Tensor>> prelu_vjp(const Tensor& x, const Tensor& alpha, const Tensor& out_grad, const std::string& data_format, const std::string& mode, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::prelu_grad<LazyTensor>(x, alpha, out_grad, data_format, mode);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> psroi_pool_vjp(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, int output_channels, float spatial_scale, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::psroi_pool_grad<LazyTensor>(x, boxes, boxes_num, out_grad, pooled_height, pooled_width, output_channels, spatial_scale);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> put_along_axis_vjp(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis, const std::string& reduce, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::put_along_axis_grad<LazyTensor>(arr, indices, out_grad, axis, reduce);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> qr_vjp(const Tensor& x, const Tensor& q, const Tensor& r, const Tensor& q_grad, const Tensor& r_grad, const std::string& mode, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::qr_grad<LazyTensor>(x, q, r, q_grad, r_grad, mode);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> real_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::real_grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> reciprocal_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::reciprocal_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> relu_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::relu_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> relu6_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::relu6_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> renorm_vjp(const Tensor& x, const Tensor& out_grad, float p, int axis, float max_norm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::renorm_grad<LazyTensor>(x, out_grad, p, axis, max_norm);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> reverse_vjp(const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::reverse<LazyTensor>(out_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> roi_align_vjp(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::roi_align_grad<LazyTensor>(x, boxes, boxes_num, out_grad, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> roi_pool_vjp(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, const Tensor& arg_max, const Tensor& out_grad, int pooled_height, int pooled_width, float spatial_scale, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::roi_pool_grad<LazyTensor>(x, boxes, boxes_num, arg_max, out_grad, pooled_height, pooled_width, spatial_scale);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> roll_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& shifts_, const std::vector<int64_t>& axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::roll_grad<LazyTensor>(x, out_grad, shifts_, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> round_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::round_grad<LazyTensor>(out_grad);
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

std::vector<std::vector<paddle::Tensor>> scatter_vjp(const Tensor& index, const Tensor& updates, const Tensor& out_grad, bool overwrite, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::scatter_grad<LazyTensor>(index, updates, out_grad, overwrite);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> scatter_nd_add_vjp(const Tensor& index, const Tensor& updates, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::scatter_nd_add_grad<LazyTensor>(index, updates, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> segment_pool_vjp(const Tensor& x, const Tensor& segment_ids, const Tensor& out, const paddle::optional<Tensor>& summed_ids, const Tensor& out_grad, const std::string& pooltype, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::segment_pool_grad<LazyTensor>(x, segment_ids, out, summed_ids, out_grad, pooltype);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> selu_vjp(const Tensor& out, const Tensor& out_grad, float scale, float alpha, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::selu_grad<LazyTensor>(out, out_grad, scale, alpha);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> send_u_recv_vjp(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& reduce_op, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::send_u_recv_grad<LazyTensor>(x, src_index, dst_index, out, dst_count, out_grad, reduce_op);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> send_ue_recv_vjp(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const paddle::optional<Tensor>& out, const paddle::optional<Tensor>& dst_count, const Tensor& out_grad, const std::string& message_op, const std::string& reduce_op, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::send_ue_recv_grad<LazyTensor>(x, y, src_index, dst_index, out, dst_count, out_grad, message_op, reduce_op);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> send_uv_vjp(const Tensor& x, const Tensor& y, const Tensor& src_index, const Tensor& dst_index, const Tensor& out_grad, const std::string& message_op, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::send_uv_grad<LazyTensor>(x, y, src_index, dst_index, out_grad, message_op);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sigmoid_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sigmoid_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sigmoid_cross_entropy_with_logits_vjp(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, const Tensor& out_grad, bool normalize, int ignore_index, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sigmoid_cross_entropy_with_logits_grad<LazyTensor>(x, label, pos_weight, out_grad, normalize, ignore_index);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sign_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::scale<LazyTensor>(out_grad, 0.0f, 0.0f, true);
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

std::vector<std::vector<paddle::Tensor>> sin_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sin_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sinh_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sinh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> slogdet_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::slogdet_grad<LazyTensor>(x, out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> softplus_vjp(const Tensor& x, const Tensor& out_grad, float beta, float threshold, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::softplus_grad<LazyTensor>(x, out_grad, beta, threshold);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> softshrink_vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::softshrink_grad<LazyTensor>(x, out_grad, threshold);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> softsign_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::softsign_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> solve_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::solve_grad<LazyTensor>(x, y, out, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> spectral_norm_vjp(const Tensor& weight, const Tensor& u, const Tensor& v, const Tensor& out_grad, int dim, int power_iters, float eps, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::spectral_norm_grad<LazyTensor>(weight, u, v, out_grad, dim, power_iters, eps);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sqrt_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sqrt_grad<LazyTensor>(out, out_grad);
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

std::vector<std::vector<paddle::Tensor>> squared_l2_norm_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::squared_l2_norm_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> squeeze_vjp(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::squeeze_grad<LazyTensor>(xshape, out_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> stack_vjp(const std::vector<Tensor>& x, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::stack_grad<LazyTensor>(x, out_grad, axis);
  vjp_res[0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> stanh_vjp(const Tensor& x, const Tensor& out_grad, float scale_a, float scale_b, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::stanh_grad<LazyTensor>(x, out_grad, scale_a, scale_b);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> svd_vjp(const Tensor& x, const Tensor& u, const Tensor& vh, const Tensor& s, const paddle::optional<Tensor>& u_grad, const paddle::optional<Tensor>& vh_grad, const paddle::optional<Tensor>& s_grad, bool full_matrices, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::svd_grad<LazyTensor>(x, u, vh, s, u_grad, vh_grad, s_grad, full_matrices);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> take_along_axis_vjp(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::take_along_axis_grad<LazyTensor>(arr, indices, out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tan_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tan_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tanh_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "tanh_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op tanh_grad";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::tanh_grad<LazyTensor>(out, out_grad, x_grad);
  } else {
    auto op_res = backend::tanh_grad<LazyTensor>(out, out_grad);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tanh_shrink_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tanh_shrink_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> temporal_shift_vjp(const Tensor& out_grad, int seg_num, float shift_ratio, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::temporal_shift_grad<LazyTensor>(out_grad, seg_num, shift_ratio, data_format);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tensor_unfold_vjp(const Tensor& input, const Tensor& out_grad, int64_t axis, int64_t size, int64_t step, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tensor_unfold_grad<LazyTensor>(input, out_grad, axis, size, step);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> thresholded_relu_vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::thresholded_relu_grad<LazyTensor>(x, out_grad, threshold);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> topk_vjp(const Tensor& x, const Tensor& indices, const Tensor& out_grad, const Tensor& k_, int axis, bool largest, bool sorted, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::topk_grad<LazyTensor>(x, indices, out_grad, k_, axis, largest, sorted);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> trace_vjp(const Tensor& x, const Tensor& out_grad, int offset, int axis1, int axis2, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::trace_grad<LazyTensor>(x, out_grad, offset, axis1, axis2);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> triangular_solve_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, bool upper, bool transpose, bool unitriangular, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::triangular_solve_grad<LazyTensor>(x, y, out, out_grad, upper, transpose, unitriangular);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> trilinear_interp_vjp(const Tensor& x, const paddle::optional<Tensor>& out_size, const paddle::optional<std::vector<Tensor>>& size_tensor, const paddle::optional<Tensor>& scale_tensor, const Tensor& output_grad, const std::string& data_layout, int out_d, int out_h, int out_w, const std::vector<float>& scale, const std::string& interp_method, bool align_corners, int align_mode, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::trilinear_interp_grad<LazyTensor>(x, out_size, size_tensor, scale_tensor, output_grad, data_layout, out_d, out_h, out_w, scale, interp_method, align_corners, align_mode);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> trunc_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::trunc_grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> unbind_vjp(const std::vector<Tensor>& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::stack<LazyTensor>(out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> unfold_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::unfold_grad<LazyTensor>(x, out_grad, kernel_sizes, strides, paddings, dilations);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> uniform_inplace_vjp(const Tensor& out_grad, float min, float max, int seed, int diag_num, int diag_step, float diag_val, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::uniform_inplace_grad<LazyTensor>(out_grad, min, max, seed, diag_num, diag_step, diag_val);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> unpool3d_vjp(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_size, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::unpool3d_grad<LazyTensor>(x, indices, out, out_grad, ksize, strides, paddings, output_size, data_format);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> unsqueeze_vjp(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::unsqueeze_grad<LazyTensor>(xshape, out_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> unstack_vjp(const std::vector<Tensor>& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::unstack_grad<LazyTensor>(out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> view_dtype_vjp(const Tensor& input, const Tensor& out_grad, DataType dtype, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::view_dtype_grad<LazyTensor>(input, out_grad, dtype);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> view_shape_vjp(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& dims, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::view_shape_grad<LazyTensor>(input, out_grad, dims);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> warpctc_vjp(const Tensor& logits, const paddle::optional<Tensor>& logits_length, const Tensor& warpctcgrad, const Tensor& loss_grad, int blank, bool norm_by_times, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::warpctc_grad<LazyTensor>(logits, logits_length, warpctcgrad, loss_grad, blank, norm_by_times);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> warprnnt_vjp(const Tensor& input, const Tensor& input_lengths, const Tensor& warprnntgrad, const Tensor& loss_grad, int blank, float fastemit_lambda, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::warprnnt_grad<LazyTensor>(input, input_lengths, warprnntgrad, loss_grad, blank, fastemit_lambda);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> weight_only_linear_vjp(const Tensor& x, const Tensor& weight, const paddle::optional<Tensor>& bias, const Tensor& weight_scale, const Tensor& out_grad, const std::string& weight_dtype, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::weight_only_linear_grad<LazyTensor>(x, weight, bias, weight_scale, out_grad, weight_dtype);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> where_vjp(const Tensor& condition, const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::where_grad<LazyTensor>(condition, x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> yolo_loss_vjp(const Tensor& x, const Tensor& gt_box, const Tensor& gt_label, const paddle::optional<Tensor>& gt_score, const Tensor& objectness_mask, const Tensor& gt_match_mask, const Tensor& loss_grad, const std::vector<int>& anchors, const std::vector<int>& anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth, float scale_x_y, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::yolo_loss_grad<LazyTensor>(x, gt_box, gt_label, gt_score, objectness_mask, gt_match_mask, loss_grad, anchors, anchor_mask, class_num, ignore_thresh, downsample_ratio, use_label_smooth, scale_x_y);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res[3][0] = std::get<3>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> add_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "add_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op add_grad";
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

std::vector<std::vector<paddle::Tensor>> amax_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::amax_grad<LazyTensor>(x, out, out_grad, axis, keepdim, reduce_all);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> amin_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::amin_grad<LazyTensor>(x, out, out_grad, axis, keepdim, reduce_all);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> assign_vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::assign<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> assign_out__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::assign_out__grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> batch_norm_vjp(const Tensor& x, const Tensor& scale, const Tensor& bias, const paddle::optional<Tensor>& mean_out, const paddle::optional<Tensor>& variance_out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::batch_norm_grad<LazyTensor>(x, scale, bias, mean_out, variance_out, saved_mean, saved_variance, reserve_space, out_grad, momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> c_embedding_vjp(const Tensor& weight, const Tensor& x, const Tensor& out_grad, int64_t start_index, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::c_embedding_grad<LazyTensor>(weight, x, out_grad, start_index);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cast_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "cast_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op cast_grad";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::cast_grad<LazyTensor>(x, out_grad, x_grad);
  } else {
    auto op_res = backend::cast<LazyTensor>(out_grad, x.dtype());
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> channel_shuffle_vjp(const Tensor& out_grad, int groups, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::channel_shuffle_grad<LazyTensor>(out_grad, groups, data_format);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> conv2d_transpose_vjp(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::conv2d_transpose_grad<LazyTensor>(x, filter, out_grad, output_size_, strides, paddings, output_padding, padding_algorithm, groups, dilations, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> deformable_conv_vjp(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const Tensor& out_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::deformable_conv_grad<LazyTensor>(x, offset, filter, mask, out_grad, strides, paddings, dilations, deformable_groups, groups, im2col_step);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res[3][0] = std::get<3>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> depthwise_conv2d_transpose_vjp(const Tensor& x, const Tensor& filter, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::depthwise_conv2d_transpose_grad<LazyTensor>(x, filter, out_grad, output_size_, strides, paddings, output_padding, padding_algorithm, groups, dilations, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> disable_check_model_nan_inf_vjp(const Tensor& out_grad, int unsetflag, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::disable_check_model_nan_inf_grad<LazyTensor>(out_grad, unsetflag);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> divide_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "divide_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op divide_grad";
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
  std::string op_name = "dropout_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op dropout_grad";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::dropout_grad<LazyTensor>(mask, out_grad, p, is_test, mode, x_grad);
  } else {
    auto op_res = backend::dropout_grad<LazyTensor>(mask, out_grad, p, is_test, mode);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> einsum_vjp(const std::vector<Tensor>& x_shape, const std::vector<Tensor>& inner_cache, const Tensor& out_grad, const std::string& equation, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::einsum_grad<LazyTensor>(x_shape, inner_cache, out_grad, equation);
  vjp_res[0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> elementwise_pow_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "elementwise_pow_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op elementwise_pow_grad";
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

std::vector<std::vector<paddle::Tensor>> enable_check_model_nan_inf_vjp(const Tensor& out_grad, int unsetflag, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::enable_check_model_nan_inf_grad<LazyTensor>(out_grad, unsetflag);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> exponential__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::zeros_like<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> frobenius_norm_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keep_dim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::frobenius_norm_grad<LazyTensor>(x, out, out_grad, axis, keep_dim, reduce_all);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fused_batch_norm_act_vjp(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fused_batch_norm_act_grad<LazyTensor>(x, scale, bias, out, saved_mean, saved_variance, reserve_space, out_grad, momentum, epsilon, act_type);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fused_bn_add_activation_vjp(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& act_type, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fused_bn_add_activation_grad<LazyTensor>(x, scale, bias, out, saved_mean, saved_variance, reserve_space, out_grad, momentum, epsilon, act_type);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res[3][0] = std::get<3>(op_res);
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

std::vector<std::vector<paddle::Tensor>> hardswish_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::hardswish_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> hsigmoid_loss_vjp(const Tensor& x, const Tensor& w, const Tensor& label, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, const paddle::optional<Tensor>& bias, const Tensor& pre_out, const Tensor& out_grad, int num_classes, bool is_sparse, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::hsigmoid_loss_grad<LazyTensor>(x, w, label, path, code, bias, pre_out, out_grad, num_classes, is_sparse);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> logsumexp_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::logsumexp_grad<LazyTensor>(x, out, out_grad, axis, keepdim, reduce_all);
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

std::vector<std::vector<paddle::Tensor>> max_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::max_grad<LazyTensor>(x, out, out_grad, axis_, keepdim, reduce_all);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> maximum_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::maximum_grad<LazyTensor>(x, y, out_grad);
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

std::vector<std::vector<paddle::Tensor>> min_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::min_grad<LazyTensor>(x, out, out_grad, axis_, keepdim, reduce_all);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> minimum_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::minimum_grad<LazyTensor>(x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> mish_vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::mish_grad<LazyTensor>(x, out_grad, threshold);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> multiply_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "multiply_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op multiply_grad";
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

std::vector<std::vector<paddle::Tensor>> norm_vjp(const Tensor& x, const Tensor& norm, const Tensor& out_grad, int axis, float epsilon, bool is_test, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::norm_grad<LazyTensor>(x, norm, out_grad, axis, epsilon, is_test);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pad_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& paddings, const Scalar& pad_value, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pad_grad<LazyTensor>(x, out_grad, paddings, pad_value);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pool2d_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pool2d_grad<LazyTensor>(x, out, out_grad, kernel_size_, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pool3d_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pool3d_grad<LazyTensor>(x, out, out_grad, kernel_size, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> prod_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const Tensor& dims_, bool keep_dim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::prod_grad<LazyTensor>(x, out, out_grad, dims_, keep_dim, reduce_all);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> repeat_interleave_vjp(const Tensor& x, const Tensor& out_grad, int repeats, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::repeat_interleave_grad<LazyTensor>(x, out_grad, repeats, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> repeat_interleave_with_tensor_index_vjp(const Tensor& x, const Tensor& repeats, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::repeat_interleave_with_tensor_index_grad<LazyTensor>(x, repeats, out_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> rnn_vjp(const Tensor& x, const std::vector<Tensor>& pre_state, const std::vector<Tensor>& weight_list, const paddle::optional<Tensor>& sequence_length, const Tensor& out, const Tensor& dropout_state_out, const Tensor& reserve, const Tensor& out_grad, const std::vector<Tensor>& state_grad, float dropout_prob, bool is_bidirec, int input_size, int hidden_size, int num_layers, const std::string& mode, int seed, bool is_test, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::rnn_grad<LazyTensor>(x, pre_state, weight_list, sequence_length, out, dropout_state_out, reserve, out_grad, state_grad, dropout_prob, is_bidirec, input_size, hidden_size, num_layers, mode, seed, is_test);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1] = std::get<1>(op_res);
  vjp_res[2] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> rrelu_vjp(const Tensor& x, const Tensor& noise, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::rrelu_grad<LazyTensor>(x, noise, out_grad);
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
  std::string op_name = "split_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op split_grad";
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

std::vector<std::vector<paddle::Tensor>> strided_slice_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const Tensor& strides_, const std::vector<int>& axes, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::strided_slice_grad<LazyTensor>(x, out_grad, starts_, ends_, strides_, axes);
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
  std::string op_name = "sum_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op sum_grad";
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

std::vector<std::vector<paddle::Tensor>> swish_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::swish_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sync_batch_norm__vjp(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sync_batch_norm_grad<LazyTensor>(x, scale, bias, saved_mean, saved_variance, reserve_space, out_grad, momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tile_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& repeat_times_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tile_grad<LazyTensor>(x, out_grad, repeat_times_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> trans_layout_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::trans_layout_grad<LazyTensor>(x, out_grad, perm);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> transpose_vjp(const Tensor& out_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "transpose_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op transpose_grad";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::transpose_grad<LazyTensor>(out_grad, perm, x_grad);
  } else {
    auto op_res = backend::transpose_grad<LazyTensor>(out_grad, perm);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tril_vjp(const Tensor& out_grad, int diagonal, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tril_grad<LazyTensor>(out_grad, diagonal);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> triu_vjp(const Tensor& out_grad, int diagonal, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::triu_grad<LazyTensor>(out_grad, diagonal);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> unpool_vjp(const Tensor& x, const Tensor& indices, const Tensor& out, const Tensor& out_grad, const Tensor& output_size_, const std::vector<int>& ksize, const std::vector<int>& strides, const std::vector<int>& padding, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::unpool_grad<LazyTensor>(x, indices, out, out_grad, output_size_, ksize, strides, padding, data_format);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> abs_grad_vjp(const Tensor& x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::abs_double_grad<LazyTensor>(x, grad_x_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> celu_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::celu_double_grad<LazyTensor>(x, grad_out, grad_x_grad, alpha);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> clip_grad_vjp(const Tensor& x, const Tensor& grad_x_grad, const Tensor& min_, const Tensor& max_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::clip_double_grad<LazyTensor>(x, grad_x_grad, min_, max_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> concat_grad_vjp(const std::vector<Tensor>& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::concat<LazyTensor>(grad_x_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> conv2d_grad_vjp(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, const std::vector<int>& dilations, int groups, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::conv2d_grad_grad<LazyTensor>(input, filter, grad_out, grad_input_grad, grad_filter_grad, strides, paddings, padding_algorithm, dilations, groups, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> conv3d_grad_vjp(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::conv3d_double_grad<LazyTensor>(input, filter, grad_out, grad_input_grad, grad_filter_grad, strides, paddings, padding_algorithm, groups, dilations, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cos_double_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cos_triple_grad<LazyTensor>(x, grad_out_forward, grad_x_grad_forward, grad_x_grad, grad_out_grad_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cos_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cos_double_grad<LazyTensor>(x, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> depthwise_conv2d_grad_vjp(const Tensor& input, const Tensor& filter, const Tensor& grad_out, const paddle::optional<Tensor>& grad_input_grad, const paddle::optional<Tensor>& grad_filter_grad, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::depthwise_conv2d_double_grad<LazyTensor>(input, filter, grad_out, grad_input_grad, grad_filter_grad, strides, paddings, padding_algorithm, groups, dilations, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> elu_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::elu_double_grad<LazyTensor>(x, grad_out, grad_x_grad, alpha);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> expand_grad_vjp(const Tensor& grad_x_grad, const Tensor& shape_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::expand<LazyTensor>(grad_x_grad, shape_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> instance_norm_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& fwd_scale, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float epsilon, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::instance_norm_double_grad<LazyTensor>(x, fwd_scale, saved_mean, saved_variance, grad_y, grad_x_grad, grad_scale_grad, grad_bias_grad, epsilon);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> leaky_relu_grad_vjp(const Tensor& x, const Tensor& grad_x_grad, float negative_slope, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::leaky_relu_double_grad<LazyTensor>(x, grad_x_grad, negative_slope);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log_double_grad<LazyTensor>(x, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pad3d_grad_vjp(const Tensor& grad_x_grad, const Tensor& paddings_, const std::string& mode, float pad_value, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pad3d_double_grad<LazyTensor>(grad_x_grad, paddings_, mode, pad_value, data_format);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pow_double_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_grad_x, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pow_triple_grad<LazyTensor>(x, grad_out, grad_grad_x, grad_x_grad, grad_grad_out_grad, y);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pow_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pow_double_grad<LazyTensor>(x, grad_out, grad_x_grad, y);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> relu_grad_vjp(const Tensor& out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::relu_double_grad<LazyTensor>(out, grad_x_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> rsqrt_grad_vjp(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::rsqrt_double_grad<LazyTensor>(out, grad_x, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sigmoid_double_grad_vjp(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_grad_x, const Tensor& grad_out_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sigmoid_triple_grad<LazyTensor>(out, fwd_grad_out, grad_grad_x, grad_out_grad, grad_grad_out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sigmoid_grad_vjp(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sigmoid_double_grad<LazyTensor>(out, fwd_grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sin_double_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sin_triple_grad<LazyTensor>(x, grad_out_forward, grad_x_grad_forward, grad_x_grad, grad_out_grad_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sin_grad_vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sin_double_grad<LazyTensor>(x, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> softplus_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float beta, float threshold, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::softplus_double_grad<LazyTensor>(x, grad_out, grad_x_grad, beta, threshold);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sqrt_grad_vjp(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sqrt_double_grad<LazyTensor>(out, grad_x, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> square_grad_vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::square_double_grad<LazyTensor>(x, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> squeeze_grad_vjp(const Tensor& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::squeeze<LazyTensor>(grad_x_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tanh_double_grad_vjp(const Tensor& out, const Tensor& grad_out_forward, const Tensor& grad_x_grad_forward, const paddle::optional<Tensor>& grad_out_new_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tanh_triple_grad<LazyTensor>(out, grad_out_forward, grad_x_grad_forward, grad_out_new_grad, grad_out_grad_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tanh_grad_vjp(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tanh_double_grad<LazyTensor>(out, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> unsqueeze_grad_vjp(const Tensor& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::unsqueeze<LazyTensor>(grad_x_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> add_double_grad_vjp(const Tensor& grad_grad_x, const Tensor& grad_grad_y, const Tensor& grad_grad_out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::add_triple_grad<LazyTensor>(grad_grad_x, grad_grad_y, grad_grad_out_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> add_grad_vjp(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::add_double_grad<LazyTensor>(y, grad_out, grad_x_grad, grad_y_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> batch_norm_grad_vjp(const Tensor& x, const Tensor& scale, const paddle::optional<Tensor>& out_mean, const paddle::optional<Tensor>& out_variance, const Tensor& saved_mean, const Tensor& saved_variance, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_scale_grad, const paddle::optional<Tensor>& grad_bias_grad, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::batch_norm_double_grad<LazyTensor>(x, scale, out_mean, out_variance, saved_mean, saved_variance, grad_out, grad_x_grad, grad_scale_grad, grad_bias_grad, momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> conv2d_transpose_grad_vjp(const Tensor& x, const Tensor& filter, const Tensor& grad_out, const Tensor& grad_x_grad, const Tensor& grad_filter_grad, const Tensor& output_size_, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::conv2d_transpose_double_grad<LazyTensor>(x, filter, grad_out, grad_x_grad, grad_filter_grad, output_size_, strides, paddings, output_padding, padding_algorithm, groups, dilations, data_format);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> divide_grad_vjp(const Tensor& y, const Tensor& out, const Tensor& grad_x, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::divide_double_grad<LazyTensor>(y, out, grad_x, grad_x_grad, grad_y_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> matmul_grad_vjp(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, bool transpose_x, bool transpose_y, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::matmul_double_grad<LazyTensor>(x, y, grad_out, grad_x_grad, grad_y_grad, transpose_x, transpose_y);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> mean_grad_vjp(const Tensor& grad_x_grad, const IntArray& axis, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::mean<LazyTensor>(grad_x_grad, axis, keepdim);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> multiply_double_grad_vjp(const Tensor& x, const Tensor& y, const Tensor& fwd_grad_out, const paddle::optional<Tensor>& fwd_grad_grad_x, const paddle::optional<Tensor>& fwd_grad_grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, const paddle::optional<Tensor>& grad_grad_out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::multiply_triple_grad<LazyTensor>(x, y, fwd_grad_out, fwd_grad_grad_x, fwd_grad_grad_y, grad_x_grad, grad_y_grad, grad_grad_out_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res[3][0] = std::get<3>(op_res);
  vjp_res[4][0] = std::get<4>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> multiply_grad_vjp(const Tensor& x, const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::multiply_double_grad<LazyTensor>(x, y, grad_out, grad_x_grad, grad_y_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pad_grad_vjp(const Tensor& grad_x_grad, const std::vector<int>& paddings, const Scalar& pad_value, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pad_double_grad<LazyTensor>(grad_x_grad, paddings, pad_value);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pool2d_grad_vjp(const Tensor& x, const Tensor& grad_x_grad, const Tensor& kernel_size_, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pool2d_double_grad<LazyTensor>(x, grad_x_grad, kernel_size_, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> reshape_grad_vjp(const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::reshape_double_grad<LazyTensor>(grad_out, grad_x_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
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

std::vector<std::vector<paddle::Tensor>> subtract_grad_vjp(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::subtract_double_grad<LazyTensor>(y, grad_out, grad_x_grad, grad_y_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sum_grad_vjp(const Tensor& grad_x_grad, const Tensor& axis_, bool keepdim, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sum<LazyTensor>(grad_x_grad, axis_, grad_x_grad.dtype(), keepdim);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tile_grad_vjp(const Tensor& grad_x_grad, const Tensor& repeat_times_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tile<LazyTensor>(grad_x_grad, repeat_times_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> transpose_grad_vjp(const Tensor& grad_x_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::transpose<LazyTensor>(grad_x_grad, perm);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> abs__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::abs_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> acos__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::acos_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> acosh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::acosh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> addmm__vjp(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha, float beta, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::addmm_grad<LazyTensor>(input, x, y, out_grad, alpha, beta);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> asin__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::asin_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> asinh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::asinh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> atan__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::atan_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> atanh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::atanh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> bce_loss__vjp(const Tensor& input, const Tensor& label, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::bce_loss_grad<LazyTensor>(input, label, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> ceil__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::ceil_grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> clip__vjp(const Tensor& x, const Tensor& out_grad, const Tensor& min_, const Tensor& max_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::clip_grad<LazyTensor>(x, out_grad, min_, max_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cos__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cos_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cosh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cosh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cross_entropy_with_softmax__vjp(const Tensor& label, const Tensor& softmax, const Tensor& loss_grad, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cross_entropy_with_softmax_grad<LazyTensor>(label, softmax, loss_grad, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cumprod__vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, int dim, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cumprod_grad<LazyTensor>(x, out, out_grad, dim);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cumsum__vjp(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool flatten, bool exclusive, bool reverse, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cumsum_grad<LazyTensor>(x, out_grad, axis_, flatten, exclusive, reverse);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> digamma__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::digamma_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> elu__vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::elu_grad<LazyTensor>(x, out, out_grad, alpha);
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

std::vector<std::vector<paddle::Tensor>> erfinv__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::erfinv_grad<LazyTensor>(out, out_grad);
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

std::vector<std::vector<paddle::Tensor>> expm1__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::expm1_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fill__vjp(const Tensor& out_grad, const Tensor& value_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fill_grad<LazyTensor>(out_grad, value_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fill_diagonal__vjp(const Tensor& out_grad, float value, int offset, bool wrap, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fill_diagonal_grad<LazyTensor>(out_grad, value, offset, wrap);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> fill_diagonal_tensor__vjp(const Tensor& out_grad, int64_t offset, int dim1, int dim2, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::fill_diagonal_tensor_grad<LazyTensor>(out_grad, offset, dim1, dim2);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> flatten__vjp(const Tensor& xshape, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::flatten_grad<LazyTensor>(xshape, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> floor__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::floor_grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> gaussian_inplace__vjp(const Tensor& out_grad, float mean, float std, int seed, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::gaussian_inplace_grad<LazyTensor>(out_grad, mean, std, seed);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> hardtanh__vjp(const Tensor& x, const Tensor& out_grad, float t_min, float t_max, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::hardtanh_grad<LazyTensor>(x, out_grad, t_min, t_max);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> i0__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::i0_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> index_add__vjp(const Tensor& index, const Tensor& add_value, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::index_add_grad<LazyTensor>(index, add_value, out_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> index_put__vjp(const Tensor& x, const std::vector<Tensor>& indices, const Tensor& value, const Tensor& out_grad, bool accumulate, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::index_put_grad<LazyTensor>(x, indices, value, out_grad, accumulate);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> leaky_relu__vjp(const Tensor& x, const Tensor& out_grad, float negative_slope, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::leaky_relu_grad<LazyTensor>(x, out_grad, negative_slope);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> lerp__vjp(const Tensor& x, const Tensor& y, const Tensor& weight, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::lerp_grad<LazyTensor>(x, y, weight, out, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> lgamma__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::lgamma_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log10__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log10_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log1p__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log1p_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log2__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log2_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> logit__vjp(const Tensor& x, const Tensor& out_grad, float eps, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::logit_grad<LazyTensor>(x, out_grad, eps);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> lu__vjp(const Tensor& x, const Tensor& out, const Tensor& pivots, const Tensor& out_grad, bool pivot, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::lu_grad<LazyTensor>(x, out, pivots, out_grad, pivot);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> polygamma__vjp(const Tensor& x, const Tensor& out_grad, int n, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::polygamma_grad<LazyTensor>(x, out_grad, n);
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

std::vector<std::vector<paddle::Tensor>> put_along_axis__vjp(const Tensor& arr, const Tensor& indices, const Tensor& out_grad, int axis, const std::string& reduce, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::put_along_axis_grad<LazyTensor>(arr, indices, out_grad, axis, reduce);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> reciprocal__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::reciprocal_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> relu__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::relu_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> renorm__vjp(const Tensor& x, const Tensor& out_grad, float p, int axis, float max_norm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::renorm_grad<LazyTensor>(x, out_grad, p, axis, max_norm);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> round__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::round_grad<LazyTensor>(out_grad);
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

std::vector<std::vector<paddle::Tensor>> scatter__vjp(const Tensor& index, const Tensor& updates, const Tensor& out_grad, bool overwrite, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::scatter_grad<LazyTensor>(index, updates, out_grad, overwrite);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sigmoid__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sigmoid_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sigmoid_cross_entropy_with_logits__vjp(const Tensor& x, const Tensor& label, const paddle::optional<Tensor>& pos_weight, const Tensor& out_grad, bool normalize, int ignore_index, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sigmoid_cross_entropy_with_logits_grad<LazyTensor>(x, label, pos_weight, out_grad, normalize, ignore_index);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sin__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sin_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sinh__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sinh_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sqrt__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sqrt_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> squeeze__vjp(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::squeeze_grad<LazyTensor>(xshape, out_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tan__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tan_grad<LazyTensor>(x, out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tanh__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "tanh_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op tanh_grad";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::tanh_grad<LazyTensor>(out, out_grad, x_grad);
  } else {
    auto op_res = backend::tanh_grad<LazyTensor>(out, out_grad);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> thresholded_relu__vjp(const Tensor& x, const Tensor& out_grad, float threshold, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::thresholded_relu_grad<LazyTensor>(x, out_grad, threshold);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> trunc__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::trunc_grad<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> uniform_inplace__vjp(const Tensor& out_grad, float min, float max, int seed, int diag_num, int diag_step, float diag_val, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::uniform_inplace_grad<LazyTensor>(out_grad, min, max, seed, diag_num, diag_step, diag_val);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> unsqueeze__vjp(const Tensor& xshape, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::unsqueeze_grad<LazyTensor>(xshape, out_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> where__vjp(const Tensor& condition, const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::where_grad<LazyTensor>(condition, x, y, out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> add__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "add_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op add_grad";
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

std::vector<std::vector<paddle::Tensor>> assign__vjp(const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::assign<LazyTensor>(out_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cast__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "cast_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op cast_grad";
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
  std::string op_name = "divide_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op divide_grad";
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
  std::string op_name = "multiply_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op multiply_grad";
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

std::vector<std::vector<paddle::Tensor>> transpose__vjp(const Tensor& out_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  std::string op_name = "transpose_grad";
  auto need_skip = paddle::prim::StaticCompositeContext::Instance().CheckSkipCompOps(op_name);
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled() && !need_skip) {
    FLAGS_tensor_operants_mode = "static";
    VLOG(4) << "Call PIR Decomposed backward op transpose_grad";
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 

    details::transpose_grad<LazyTensor>(out_grad, perm, x_grad);
  } else {
    auto op_res = backend::transpose_grad<LazyTensor>(out_grad, perm);
    vjp_res[0][0] = op_res;
    vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  }
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tril__vjp(const Tensor& out_grad, int diagonal, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tril_grad<LazyTensor>(out_grad, diagonal);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> triu__vjp(const Tensor& out_grad, int diagonal, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::triu_grad<LazyTensor>(out_grad, diagonal);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> celu_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::celu_double_grad<LazyTensor>(x, grad_out, grad_x_grad, alpha);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> clip_grad__vjp(const Tensor& x, const Tensor& grad_x_grad, const Tensor& min_, const Tensor& max_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::clip_double_grad<LazyTensor>(x, grad_x_grad, min_, max_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cos_double_grad__vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cos_triple_grad<LazyTensor>(x, grad_out_forward, grad_x_grad_forward, grad_x_grad, grad_out_grad_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> cos_grad__vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::cos_double_grad<LazyTensor>(x, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> elu_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float alpha, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::elu_double_grad<LazyTensor>(x, grad_out, grad_x_grad, alpha);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> leaky_relu_grad__vjp(const Tensor& x, const Tensor& grad_x_grad, float negative_slope, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::leaky_relu_double_grad<LazyTensor>(x, grad_x_grad, negative_slope);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> log_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::log_double_grad<LazyTensor>(x, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pow_double_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_grad_x, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pow_triple_grad<LazyTensor>(x, grad_out, grad_grad_x, grad_x_grad, grad_grad_out_grad, y);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> pow_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::pow_double_grad<LazyTensor>(x, grad_out, grad_x_grad, y);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> relu_grad__vjp(const Tensor& out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::relu_double_grad<LazyTensor>(out, grad_x_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> rsqrt_grad__vjp(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::rsqrt_double_grad<LazyTensor>(out, grad_x, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sigmoid_double_grad__vjp(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_grad_x, const Tensor& grad_out_grad, const paddle::optional<Tensor>& grad_grad_out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sigmoid_triple_grad<LazyTensor>(out, fwd_grad_out, grad_grad_x, grad_out_grad, grad_grad_out_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sigmoid_grad__vjp(const Tensor& out, const Tensor& fwd_grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sigmoid_double_grad<LazyTensor>(out, fwd_grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sin_double_grad__vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out_forward, const paddle::optional<Tensor>& grad_x_grad_forward, const Tensor& grad_x_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sin_triple_grad<LazyTensor>(x, grad_out_forward, grad_x_grad_forward, grad_x_grad, grad_out_grad_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sin_grad__vjp(const Tensor& x, const paddle::optional<Tensor>& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sin_double_grad<LazyTensor>(x, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> softplus_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, float beta, float threshold, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::softplus_double_grad<LazyTensor>(x, grad_out, grad_x_grad, beta, threshold);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> sqrt_grad__vjp(const Tensor& out, const Tensor& grad_x, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::sqrt_double_grad<LazyTensor>(out, grad_x, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> square_grad__vjp(const Tensor& x, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::square_double_grad<LazyTensor>(x, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> squeeze_grad__vjp(const Tensor& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::squeeze<LazyTensor>(grad_x_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tanh_double_grad__vjp(const Tensor& out, const Tensor& grad_out_forward, const Tensor& grad_x_grad_forward, const paddle::optional<Tensor>& grad_out_new_grad, const paddle::optional<Tensor>& grad_out_grad_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tanh_triple_grad<LazyTensor>(out, grad_out_forward, grad_x_grad_forward, grad_out_new_grad, grad_out_grad_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> tanh_grad__vjp(const Tensor& out, const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tanh_double_grad<LazyTensor>(out, grad_out, grad_x_grad);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> unsqueeze_grad__vjp(const Tensor& grad_x_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::unsqueeze<LazyTensor>(grad_x_grad, axis_);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> add_double_grad__vjp(const Tensor& grad_grad_x, const Tensor& grad_grad_y, const Tensor& grad_grad_out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::add_triple_grad<LazyTensor>(grad_grad_x, grad_grad_y, grad_grad_out_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> add_grad__vjp(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::add_double_grad<LazyTensor>(y, grad_out, grad_x_grad, grad_y_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> multiply_double_grad__vjp(const Tensor& x, const Tensor& y, const Tensor& fwd_grad_out, const paddle::optional<Tensor>& fwd_grad_grad_x, const paddle::optional<Tensor>& fwd_grad_grad_y, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, const paddle::optional<Tensor>& grad_grad_out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::multiply_triple_grad<LazyTensor>(x, y, fwd_grad_out, fwd_grad_grad_x, fwd_grad_grad_y, grad_x_grad, grad_y_grad, grad_grad_out_grad, axis);
  vjp_res[0][0] = std::get<0>(op_res);
  vjp_res[1][0] = std::get<1>(op_res);
  vjp_res[2][0] = std::get<2>(op_res);
  vjp_res[3][0] = std::get<3>(op_res);
  vjp_res[4][0] = std::get<4>(op_res);
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> reshape_grad__vjp(const Tensor& grad_out, const Tensor& grad_x_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::reshape_double_grad<LazyTensor>(grad_out, grad_x_grad);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}

std::vector<std::vector<paddle::Tensor>> subtract_grad__vjp(const Tensor& y, const Tensor& grad_out, const paddle::optional<Tensor>& grad_x_grad, const paddle::optional<Tensor>& grad_y_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::subtract_double_grad<LazyTensor>(y, grad_out, grad_x_grad, grad_y_grad, axis);
  vjp_res[0][0] = op_res;
  vjp_res = ConstructVjpResultByStopGradients(vjp_res, stop_gradients);
  return vjp_res;
}



}  // namespace primitive
}  // namespace paddle
