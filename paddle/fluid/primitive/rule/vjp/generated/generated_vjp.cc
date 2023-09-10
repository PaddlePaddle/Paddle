// Auto Generated, DO NOT EDIT!

#include "paddle/fluid/primitive/rule/vjp/generated/generated_vjp.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_api.h"
#include "paddle/fluid/prim/utils/static/static_global_utils.h"
#include "paddle/fluid/primitive/backend/backend.h"
#include "paddle/fluid/primitive/rule/vjp/details.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"
#include "paddle/ir/core/operation.h"


namespace paddle {
namespace primitive {


std::vector<std::vector<paddle::Tensor>> tanh_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::tanh_grad<LazyTensor>(out, out_grad);
  vjp_res[0][0] = !stop_gradients[0][0] ? op_res : vjp_res[0][0];

  return vjp_res;

}

std::vector<std::vector<paddle::Tensor>> add_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::add_grad<LazyTensor>(x, y, out_grad, axis);
  auto out0 = std::get<0>(op_res);
  vjp_res[0][0] = !stop_gradients[0][0] ? out0 : vjp_res[0][0];
  auto out1 = std::get<1>(op_res);
  vjp_res[1][0] = !stop_gradients[1][0] ? out1 : vjp_res[1][0];

  return vjp_res;

}

std::vector<std::vector<paddle::Tensor>> divide_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    paddle::Tensor* y_grad = !stop_gradients[1][0] ? &vjp_res[1][0] : nullptr; 
    details::divide_grad<LazyTensor>(x, y, out, out_grad, axis, x_grad, y_grad);
  } else {
    auto op_res = backend::divide_grad<LazyTensor>(x, y, out, out_grad, axis);
    auto out0 = std::get<0>(op_res);
    vjp_res[0][0] = !stop_gradients[0][0] ? out0 : vjp_res[0][0];
    auto out1 = std::get<1>(op_res);
    vjp_res[1][0] = !stop_gradients[1][0] ? out1 : vjp_res[1][0];
  }
  return vjp_res;

}

std::vector<std::vector<paddle::Tensor>> mean_vjp(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  auto op_res = backend::mean_grad<LazyTensor>(x, out_grad, axis, keepdim, reduce_all);
  vjp_res[0][0] = !stop_gradients[0][0] ? op_res : vjp_res[0][0];

  return vjp_res;

}

std::vector<std::vector<paddle::Tensor>> sum_vjp(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients) {
  std::vector<std::vector<paddle::Tensor>> vjp_res;
  for (auto arg: stop_gradients) {
    vjp_res.push_back(std::vector<paddle::Tensor>(arg.size()));
  }
  if (paddle::prim::StaticCompositeContext::Instance().IsBwdPrimEnabled()) {
    paddle::Tensor* x_grad = !stop_gradients[0][0] ? &vjp_res[0][0] : nullptr; 
    details::sum_grad<LazyTensor>(x, out_grad, axis, keepdim, reduce_all, x_grad);
  } else {
    auto op_res = backend::sum_grad<LazyTensor>(x, out_grad, axis, keepdim, reduce_all);
    vjp_res[0][0] = !stop_gradients[0][0] ? op_res : vjp_res[0][0];
  }
  return vjp_res;

}



}  // namespace primitive
}  // namespace paddle
