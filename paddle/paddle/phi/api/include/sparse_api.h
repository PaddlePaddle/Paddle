#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {
namespace sparse {


// out

PADDLE_API Tensor abs(const Tensor& x);


// out

PADDLE_API Tensor acos(const Tensor& x);


// out

PADDLE_API Tensor acosh(const Tensor& x);


// out

PADDLE_API Tensor add(const Tensor& x, const Tensor& y);


// out

PADDLE_API Tensor asin(const Tensor& x);


// out

PADDLE_API Tensor asinh(const Tensor& x);


// out

PADDLE_API Tensor atan(const Tensor& x);


// out

PADDLE_API Tensor atanh(const Tensor& x);


// out, mean_out, variance_out, saved_mean, saved_variance, reserve_space

PADDLE_API std::tuple<Tensor, Tensor&, Tensor&, Tensor, Tensor, Tensor> batch_norm_(const Tensor& x, Tensor& mean, Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_format, bool use_global_stats, bool trainable_statistics);


// out

PADDLE_API Tensor cast(const Tensor& x, DataType index_dtype = DataType::UNDEFINED, DataType value_dtype = DataType::UNDEFINED);


// out, rulebook, counter

PADDLE_API Tensor conv3d(const Tensor& x, const Tensor& kernel, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key = "");


// out

PADDLE_API Tensor divide(const Tensor& x, const Tensor& y);


// out

PADDLE_API Tensor divide_scalar(const Tensor& x, float scalar);


// out

PADDLE_API Tensor expm1(const Tensor& x);


// out

PADDLE_API Tensor isnan(const Tensor& x);


// out

PADDLE_API Tensor leaky_relu(const Tensor& x, float alpha);


// out

PADDLE_API Tensor log1p(const Tensor& x);


// out

PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y);


// out

PADDLE_API Tensor pow(const Tensor& x, float factor);


// out

PADDLE_API Tensor relu(const Tensor& x);


// out

PADDLE_API Tensor relu6(const Tensor& x);


// out

PADDLE_API Tensor reshape(const Tensor& x, const IntArray& shape);


// out

PADDLE_API Tensor scale(const Tensor& x, float scale, float bias, bool bias_after_scale);


// out

PADDLE_API Tensor sin(const Tensor& x);


// out

PADDLE_API Tensor sinh(const Tensor& x);


// out

PADDLE_API Tensor softmax(const Tensor& x, int axis = -1);


// out

PADDLE_API Tensor sparse_coo_tensor(const Tensor& values, const Tensor& indices, const std::vector<int64_t>& shape = {});


// out

PADDLE_API Tensor sqrt(const Tensor& x);


// out

PADDLE_API Tensor square(const Tensor& x);


// out

PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y);


// out

PADDLE_API Tensor sum(const Tensor& x, const IntArray& axis = {}, DataType dtype = DataType::UNDEFINED, bool keepdim = false);


// out, mean_out, variance_out, saved_mean, saved_variance, reserve_space

PADDLE_API std::tuple<Tensor, Tensor&, Tensor&, Tensor, Tensor, Tensor> sync_batch_norm_(const Tensor& x, Tensor& mean, Tensor& variance, const Tensor& scale, const Tensor& bias, bool is_test, float momentum, float epsilon, const std::string& data_format, bool use_global_stats, bool trainable_statistics);


// out

PADDLE_API Tensor tan(const Tensor& x);


// out

PADDLE_API Tensor tanh(const Tensor& x);


// out

PADDLE_API Tensor to_dense(const Tensor& x);


// out

PADDLE_API Tensor to_sparse_coo(const Tensor& x, int64_t sparse_dim);


// out

PADDLE_API Tensor to_sparse_csr(const Tensor& x);


// out

PADDLE_API Tensor transpose(const Tensor& x, const std::vector<int>& perm);


// out

PADDLE_API Tensor values(const Tensor& x);


// out

PADDLE_API Tensor addmm(const Tensor& input, const Tensor& x, const Tensor& y, float beta = 1.0, float alpha = 1.0);


// out

PADDLE_API Tensor coalesce(const Tensor& x);


// out

PADDLE_API Tensor full_like(const Tensor& x, const Scalar& value, DataType dtype = DataType::UNDEFINED);


// out, softmax

PADDLE_API Tensor fused_attention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& sparse_mask, const paddle::optional<Tensor>& key_padding_mask, const paddle::optional<Tensor>& attn_mask);


// out

PADDLE_API Tensor masked_matmul(const Tensor& x, const Tensor& y, const Tensor& mask);


// out

PADDLE_API Tensor matmul(const Tensor& x, const Tensor& y);


// out, rulebook, counter

PADDLE_API Tensor maxpool(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides);


// out

PADDLE_API Tensor mv(const Tensor& x, const Tensor& vec);


// out

PADDLE_API Tensor slice(const Tensor& x, const IntArray& axes, const IntArray& starts, const IntArray& ends);



}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
