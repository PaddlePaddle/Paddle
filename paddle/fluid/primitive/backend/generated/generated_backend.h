// Auto Generated, DO NOT EDIT!

#pragma once

#include <string>
#include <vector>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/utils/optional.h"


namespace paddle {
namespace primitive {
namespace backend {

using Tensor = paddle::Tensor;
using Scalar = paddle::experimental::Scalar;
using IntArray = paddle::experimental::IntArray;
using DataType = phi::DataType;

template <typename T>
Tensor abs(const Tensor& x);

template <typename T>
Tensor bitwise_and(const Tensor& x, const Tensor& y);

template <typename T>
Tensor bitwise_not(const Tensor& x);

template <typename T>
Tensor bitwise_or(const Tensor& x, const Tensor& y);

template <typename T>
Tensor bitwise_xor(const Tensor& x, const Tensor& y);

template <typename T>
Tensor concat(const std::vector<Tensor>& x, const Tensor& axis_);

template <typename T>
Tensor concat(const std::vector<Tensor>& x, const Scalar& axis = 0);

template <typename T>
Tensor erf(const Tensor& x);

template <typename T>
Tensor exp(const Tensor& x);

template <typename T>
Tensor expand(const Tensor& x, const Tensor& shape_);

template <typename T>
Tensor expand(const Tensor& x, const IntArray& shape = {});

template <typename T>
Tensor floor(const Tensor& x);

template <typename T>
Tensor gather_nd(const Tensor& x, const Tensor& index);

template <typename T>
Tensor log(const Tensor& x);

template <typename T>
Tensor roll(const Tensor& x, const Tensor& shifts_, const std::vector<int64_t>& axis = {});

template <typename T>
Tensor roll(const Tensor& x, const IntArray& shifts = {}, const std::vector<int64_t>& axis = {});

template <typename T>
Tensor scale(const Tensor& x, const Tensor& scale_, float bias = 0.0, bool bias_after_scale = true);

template <typename T>
Tensor scale(const Tensor& x, const Scalar& scale = 1.0, float bias = 0.0, bool bias_after_scale = true);

template <typename T>
Tensor scatter(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite = true);

template <typename T>
Tensor scatter_nd_add(const Tensor& x, const Tensor& index, const Tensor& updates);

template <typename T>
Tensor sqrt(const Tensor& x);

template <typename T>
Tensor tanh(const Tensor& x);

template <typename T>
Tensor add(const Tensor& x, const Tensor& y);

template <typename T>
Tensor add_n(const std::vector<Tensor>& inputs);

template <typename T>
Tensor assign(const Tensor& x);

template <typename T>
Tensor cast(const Tensor& x, DataType dtype);

template <typename T>
Tensor divide(const Tensor& x, const Tensor& y);

template <typename T>
Tensor elementwise_pow(const Tensor& x, const Tensor& y);

template <typename T>
Tensor equal(const Tensor& x, const Tensor& y);

template <typename T>
Tensor full(const IntArray& shape, const Scalar& value, DataType dtype = DataType::FLOAT32, Place place = CPUPlace());

template <typename T>
Tensor greater_equal(const Tensor& x, const Tensor& y);

template <typename T>
Tensor greater_than(const Tensor& x, const Tensor& y);

template <typename T>
Tensor less_equal(const Tensor& x, const Tensor& y);

template <typename T>
Tensor less_than(const Tensor& x, const Tensor& y);

template <typename T>
Tensor matmul(const Tensor& x, const Tensor& y, bool transpose_x = false, bool transpose_y = false);

template <typename T>
Tensor max(const Tensor& x, const Tensor& axis_, bool keepdim = false);

template <typename T>
Tensor max(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

template <typename T>
Tensor maximum(const Tensor& x, const Tensor& y);

template <typename T>
Tensor mean(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

template <typename T>
Tensor minimum(const Tensor& x, const Tensor& y);

template <typename T>
Tensor multiply(const Tensor& x, const Tensor& y);

template <typename T>
Tensor not_equal(const Tensor& x, const Tensor& y);

template <typename T>
Tensor reshape(const Tensor& x, const Tensor& shape_);

template <typename T>
Tensor reshape(const Tensor& x, const IntArray& shape);

template <typename T>
Tensor slice(const Tensor& input, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

template <typename T>
Tensor slice(const Tensor& input, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

template <typename T>
std::vector<Tensor> split(const Tensor& x, const Tensor& sections_, const Tensor& axis_);

template <typename T>
std::vector<Tensor> split(const Tensor& x, const IntArray& sections, const Scalar& axis);

template <typename T>
Tensor subtract(const Tensor& x, const Tensor& y);

template <typename T>
Tensor sum(const Tensor& x, const Tensor& axis_, DataType dtype = DataType::UNDEFINED, bool keepdim = false);

template <typename T>
Tensor sum(const Tensor& x, const IntArray& axis = {}, DataType dtype = DataType::UNDEFINED, bool keepdim = false);

template <typename T>
Tensor tile(const Tensor& x, const Tensor& repeat_times_);

template <typename T>
Tensor tile(const Tensor& x, const IntArray& repeat_times = {});

template <typename T>
Tensor transpose(const Tensor& x, const std::vector<int>& perm);

template <typename T>
Tensor uniform(const Tensor& shape_, const Tensor& min_, const Tensor& max_, DataType dtype, int seed, Place place = {});

template <typename T>
Tensor uniform(const IntArray& shape, DataType dtype, const Scalar& min, const Scalar& max, int seed, Place place = {});

template <typename T>
std::vector<Tensor> concat_grad(const std::vector<Tensor>& x, const Tensor& out_grad, const Tensor& axis_);

template <typename T>
std::vector<Tensor> concat_grad(const std::vector<Tensor>& x, const Tensor& out_grad, const Scalar& axis = 0);

template <typename T>
Tensor erf_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor exp_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor expand_grad(const Tensor& x, const Tensor& out_grad, const Tensor& shape_);

template <typename T>
Tensor expand_grad(const Tensor& x, const Tensor& out_grad, const IntArray& shape);

template <typename T>
Tensor gelu_grad(const Tensor& x, const Tensor& out_grad, bool approximate);

template <typename T>
std::tuple<Tensor, Tensor, Tensor> layer_norm_grad(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, const Tensor& mean, const Tensor& variance, const Tensor& out_grad, float epsilon = 1e-5, int begin_norm_axis = 1);

template <typename T>
Tensor pow_grad(const Tensor& x, const Tensor& out_grad, const Scalar& y = -1);

template <typename T>
Tensor rsqrt_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor silu_grad(const Tensor& x, const Tensor& out, const Tensor& out_grad);

template <typename T>
Tensor square_grad(const Tensor& x, const Tensor& out_grad);

template <typename T>
Tensor tanh_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> add_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> divide_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis = -1);

template <typename T>
Tensor dropout_grad(const Tensor& mask, const Tensor& out_grad, const Scalar& p, bool is_test, const std::string& mode);

template <typename T>
std::tuple<Tensor, Tensor> elementwise_pow_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);

template <typename T>
Tensor embedding_grad(const Tensor& x, const Tensor& weight, const Tensor& out_grad, int64_t padding_idx = -1, bool sparse = false);

template <typename T>
Tensor fused_softmax_mask_upper_triangle_grad(const Tensor& Out, const Tensor& Out_grad);

template <typename T>
std::tuple<Tensor, Tensor> matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, bool transpose_x = false, bool transpose_y = false);

template <typename T>
Tensor mean_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis = {}, bool keepdim = false, bool reduce_all = false);

template <typename T>
std::tuple<Tensor, Tensor> multiply_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis = -1);

template <typename T>
Tensor reshape_grad(const Tensor& xshape, const Tensor& out_grad);

template <typename T>
Tensor slice_grad(const Tensor& input, const Tensor& out_grad, const Tensor& starts_, const Tensor& ends_, const std::vector<int64_t>& axes, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

template <typename T>
Tensor slice_grad(const Tensor& input, const Tensor& out_grad, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis);

template <typename T>
Tensor softmax_grad(const Tensor& out, const Tensor& out_grad, int axis);

template <typename T>
Tensor split_grad(const std::vector<Tensor>& out_grad, const Tensor& axis_);

template <typename T>
Tensor split_grad(const std::vector<Tensor>& out_grad, const Scalar& axis = -1);

template <typename T>
Tensor split_with_num_grad(const std::vector<Tensor>& out_grad, const Tensor& axis_);

template <typename T>
Tensor split_with_num_grad(const std::vector<Tensor>& out_grad, const Scalar& axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> subtract_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis = -1);

template <typename T>
Tensor sum_grad(const Tensor& x, const Tensor& out_grad, const Tensor& axis_, bool keepdim, bool reduce_all = false);

template <typename T>
Tensor sum_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all = false);

template <typename T>
Tensor transpose_grad(const Tensor& out_grad, const std::vector<int>& perm);

}  // namespace backend
}  // namespace primitive
}  // namespace paddle
