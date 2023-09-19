// Auto Generated, DO NOT EDIT!

#pragma once

#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/pir/core/value.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"

namespace paddle {
namespace primitive {

using IntArray = paddle::experimental::IntArray;

std::vector<std::vector<paddle::Tensor>> concat_vjp(const std::vector<Tensor>& x, const Tensor& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> erf_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> exp_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> expand_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& shape_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> gelu_vjp(const Tensor& x, const Tensor& out_grad, bool approximate, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> layer_norm_vjp(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& mean, const Tensor& variance, const Tensor& out_grad, float epsilon, int begin_norm_axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pow_vjp(const Tensor& x, const Tensor& out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> rsqrt_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> scale_vjp(const Tensor& out_grad, const Tensor& scale_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> silu_vjp(const Tensor& x, const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> square_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tanh_vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> add_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cast_vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> divide_vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> dropout_vjp(const Tensor& mask, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> elementwise_pow_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> embedding_vjp(const Tensor& x, const Tensor& weight, const Tensor& out_grad, int64_t padding_idx, bool sparse, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> fused_softmax_mask_upper_triangle_vjp(const Tensor& Out, const Tensor& Out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> matmul_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x, bool transpose_y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> mean_vjp(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> multiply_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> reshape_vjp(const Tensor& xshape, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> slice_vjp(const Tensor& input, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> softmax_vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> split_vjp(const std::vector<Tensor>& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> split_with_num_vjp(const std::vector<Tensor>& out_grad, const Tensor& axis_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> subtract_vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> sum_vjp(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> transpose_vjp(const Tensor& out_grad, const std::vector<int>& perm, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> slice_grad_vjp(const Tensor& grad_input_grad, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> erf__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> exp__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> pow__vjp(const Tensor& x, const Tensor& out_grad, const Scalar& y, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> rsqrt__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> scale__vjp(const Tensor& out_grad, const Tensor& scale_, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> tanh__vjp(const Tensor& out, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> add__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> cast__vjp(const Tensor& x, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> divide__vjp(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> multiply__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> reshape__vjp(const Tensor& xshape, const Tensor& out_grad, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> softmax__vjp(const Tensor& out, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


std::vector<std::vector<paddle::Tensor>> subtract__vjp(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis, const std::vector<std::vector<bool>>& stop_gradients);


}  // namespace primitive
}  // namespace paddle
