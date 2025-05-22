#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {
namespace sparse {


// x_grad

PADDLE_API Tensor abs_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor acos_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor acosh_grad(const Tensor& x, const Tensor& out_grad);


// x_grad, y_grad

PADDLE_API std::tuple<Tensor, Tensor> add_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);


// input_grad, x_grad, y_grad

PADDLE_API std::tuple<Tensor, Tensor, Tensor> addmm_grad(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha = 1.0, float beta = 1.0);


// x_grad

PADDLE_API Tensor asin_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor asinh_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor atan_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor atanh_grad(const Tensor& x, const Tensor& out_grad);


// x_grad, scale_grad, bias_grad

PADDLE_API std::tuple<Tensor, Tensor, Tensor> batch_norm_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const paddle::optional<Tensor>& mean_out, const paddle::optional<Tensor>& variance_out, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_format, bool is_test, bool use_global_stats, bool trainable_statistics);


// x_grad

PADDLE_API Tensor cast_grad(const Tensor& x, const Tensor& out_grad, DataType value_dtype);


// x_grad, kernel_grad

PADDLE_API std::tuple<Tensor, Tensor> conv3d_grad(const Tensor& x, const Tensor& kernel, const Tensor& out, const Tensor& rulebook, const Tensor& counter, const Tensor& out_grad, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key);


// x_grad, y_grad

PADDLE_API std::tuple<Tensor, Tensor> divide_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor divide_scalar_grad(const Tensor& out_grad, float scalar);


// x_grad

PADDLE_API Tensor expm1_grad(const Tensor& out, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor leaky_relu_grad(const Tensor& x, const Tensor& out_grad, float alpha);


// x_grad

PADDLE_API Tensor log1p_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor mask_as_grad(const Tensor& x, const Tensor& mask, const Tensor& out_grad);


// x_grad, y_grad

PADDLE_API std::tuple<Tensor, Tensor> masked_matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);


// x_grad, y_grad

PADDLE_API std::tuple<Tensor, Tensor> matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor maxpool_grad(const Tensor& x, const Tensor& rulebook, const Tensor& counter, const Tensor& out, const Tensor& out_grad, const std::vector<int>& kernel_sizes);


// x_grad, y_grad

PADDLE_API std::tuple<Tensor, Tensor> multiply_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);


// x_grad, vec_grad

PADDLE_API std::tuple<Tensor, Tensor> mv_grad(const Tensor& x, const Tensor& vec, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor pow_grad(const Tensor& x, const Tensor& out_grad, float factor);


// x_grad

PADDLE_API Tensor relu6_grad(const Tensor& out, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor relu_grad(const Tensor& out, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor reshape_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor sin_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor sinh_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor softmax_grad(const Tensor& out, const Tensor& out_grad, int axis);


// values_grad

PADDLE_API Tensor sparse_coo_tensor_grad(const Tensor& indices, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor sqrt_grad(const Tensor& out, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor square_grad(const Tensor& x, const Tensor& out_grad);


// x_grad, y_grad

PADDLE_API std::tuple<Tensor, Tensor> subtract_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor sum_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis = {}, bool keepdim = false);


// x_grad, scale_grad, bias_grad

PADDLE_API std::tuple<Tensor, Tensor, Tensor> sync_batch_norm_grad(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& saved_mean, const Tensor& saved_variance, const paddle::optional<Tensor>& reserve_space, const Tensor& out_grad, float momentum, float epsilon, const std::string& data_format, bool is_test, bool use_global_stats, bool trainable_statistics);


// x_grad

PADDLE_API Tensor tan_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor tanh_grad(const Tensor& out, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor to_dense_grad(const Tensor& x, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor to_sparse_coo_grad(const Tensor& out_grad);


// x_grad

PADDLE_API Tensor transpose_grad(const Tensor& out_grad, const std::vector<int>& perm);


// x_grad

PADDLE_API Tensor values_grad(const Tensor& x, const Tensor& out_grad);


// query_grad, key_grad, value_grad

PADDLE_API std::tuple<Tensor, Tensor, Tensor> fused_attention_grad(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& softmax, const Tensor& out_grad);


// x_grad

PADDLE_API Tensor slice_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axes, const IntArray& starts, const IntArray& ends);



}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
