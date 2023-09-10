// Auto Generated, DO NOT EDIT!

#pragma once

#include <string>
#include <vector>

#include "paddle/phi/api/include/tensor.h"


namespace paddle {
namespace primitive {
namespace backend {

using Tensor = paddle::Tensor;
using Scalar = paddle::experimental::Scalar;
using IntArray = paddle::experimental::IntArray;
using DataType = phi::DataType;

template <typename T>
Tensor concat(const std::vector<Tensor>& x, const Scalar& axis = 0);

template <typename T>
Tensor expand(const Tensor& x, const IntArray& shape = {});

template <typename T>
Tensor scale(const Tensor& x, const Scalar& scale = 1.0, float bias = 0.0, bool bias_after_scale = true);

template <typename T>
Tensor add(const Tensor& x, const Tensor& y);

template <typename T>
Tensor add_n(const std::vector<Tensor>& inputs);

template <typename T>
Tensor divide(const Tensor& x, const Tensor& y);

template <typename T>
Tensor elementwise_pow(const Tensor& x, const Tensor& y);

template <typename T>
Tensor full(const IntArray& shape, const Scalar& value, DataType dtype = DataType::FLOAT32, Place place = CPUPlace());

template <typename T>
Tensor mean(const Tensor& x, const IntArray& axis = {}, bool keepdim = false);

template <typename T>
Tensor multiply(const Tensor& x, const Tensor& y);

template <typename T>
std::tuple<Tensor, Tensor> reshape(const Tensor& x, const IntArray& shape);

template <typename T>
Tensor sum(const Tensor& x, const IntArray& axis = {}, DataType dtype = DataType::UNDEFINED, bool keepdim = false);

template <typename T>
Tensor tile(const Tensor& x, const IntArray& repeat_times = {});

template <typename T>
Tensor tanh_grad(const Tensor& out, const Tensor& out_grad);

template <typename T>
std::tuple<Tensor, Tensor> add_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, int axis = -1);

template <typename T>
std::tuple<Tensor, Tensor> divide_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, int axis = -1);

template <typename T>
Tensor mean_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis = {}, bool keepdim = false, bool reduce_all = false);

template <typename T>
Tensor sum_grad(const Tensor& x, const Tensor& out_grad, const IntArray& axis, bool keepdim, bool reduce_all = false);

}  // namespace backend
}  // namespace primitive
}  // namespace paddle
