// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/fluid/prim/api/all.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace prim {
using Tensor = paddle::experimental::Tensor;
using IntArray =
    paddle::experimental::IntArrayBase<paddle::experimental::Tensor>;
//  This function should have as same signature as phi, which defined in
//  paddle/phi/api/backward/backward_api.h
template <typename T>
void gather_grad(const Tensor& x,
                 const Tensor& index,
                 const Tensor& out_grad,
                 const Scalar& axis,
                 bool overwrite,
                 Tensor* grad_x) {
  auto zero_tensor = full<T>(phi::vectorize(x.dims()), 0.0, x.dtype());
  std::vector<int> tmp_perm;

  // change axis to rank 0
  int axis_value = axis.to<int>();
  tmp_perm.push_back(axis_value);
  // make other ranks
  for (int i = 0; i < x.dims().size(); ++i) {
    if (i != axis_value) {
      tmp_perm.push_back(i);
    }
  }
  std::vector<int> reverse_perm(tmp_perm);
  // make origin ranks
  for (int i = 0; i < static_cast<int>(tmp_perm.size()); ++i) {
    reverse_perm[tmp_perm[i]] = i;
  }

  // transpose out_grad and zero grad to target rank.
  auto tmp_zero_x_grad = transpose<T>(zero_tensor, tmp_perm);
  auto tmp_out_grad = transpose<T>(out_grad, tmp_perm);
  // scatter grad to grad_x
  auto tmp_grad_x = scatter<T>(tmp_zero_x_grad, index, tmp_out_grad, false);
  auto tmp_grad_x_tranposed = transpose<T>(tmp_grad_x, reverse_perm);
  set_output<T>(tmp_grad_x_tranposed, grad_x);
}

template <typename T>
void tanh_grad(const Tensor& out, const Tensor& grad_out, Tensor* grad_x) {
  if (!grad_x) return;
  auto grad_x_tmp = grad_out * (1.0 - out.pow(2.0));
  set_output<T>(grad_x_tmp, grad_x);
}

template <typename T>
void subtract_grad(const Tensor& x,
                   const Tensor& y,
                   const Tensor& out_grad,
                   int axis,
                   Tensor* dx,
                   Tensor* dy) {
  if (dy) {
    auto scale_out_grad = scale<T>(out_grad, -1.0, 0.0, true);
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(scale_out_grad, dy);
      } else {
        auto dy_reduce_res =
            scale_out_grad.sum(phi::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      }
    } else {
      by_pass<T>(scale_out_grad, dy);
    }
  }
  if (dx) {
    if (y.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), y.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(out_grad, dx);
      } else {
        auto dx_reduce_res =
            out_grad.sum(phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      }
    } else {
      by_pass<T>(out_grad, dx);
    }
  }
}

template <typename T>
void add_grad(const Tensor& x,
              const Tensor& y,
              const Tensor& out_grad,
              int axis,
              Tensor* dx,
              Tensor* dy) {
  if (dy) {
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(out_grad, dy);
      } else {
        auto dy_reduce_res =
            out_grad.sum(phi::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      }

    } else {
      by_pass<T>(out_grad, dy);
    }
  }
  if (dx) {
    if (y.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), y.dims());
      if (!reduce_dim.size()) {
        by_pass<T>(out_grad, dx);
      } else {
        auto dx_reduce_res =
            out_grad.sum(phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      }
    } else {
      by_pass<T>(out_grad, dx);
    }
  }
}

template <typename T>
void sum_grad(const Tensor& x,
              const Tensor& out_grad,
              const IntArray& axis,
              bool keepdim,
              bool reduce_all,
              Tensor* x_grad) {
  if (!x_grad) {
    return;
  }
  std::vector<int64_t> x_dim = phi::vectorize<int64_t>(x.dims());
  int64_t axis_size = axis.size();
  int64_t x_dim_size = x_dim.size();
  reduce_all = false;
  if (reduce_all || axis_size == 0 || axis_size == x_dim_size) {
    reduce_all = true;
  } else {
    reduce_all = false;
  }
  auto x_grad_tmp = Tensor();
  if (x_dim_size == 1) {
    x_grad_tmp = out_grad.expand(IntArray(x_dim));
  } else {
    if (!keepdim) {
      auto axis_ = std::vector<int64_t>();
      if (reduce_all) {
        for (int64_t i = 1; i < x_dim_size; i++) {
          axis_.push_back(i);
        }
      } else {
        axis_ = axis.GetData();
      }
      auto out_grad_ = unsqueeze<T>(out_grad, axis_);
      x_grad_tmp = out_grad_.expand(IntArray(x_dim));
    } else {
      x_grad_tmp = out_grad.expand(IntArray(x_dim));
    }
  }

  set_output<T>(x_grad_tmp, x_grad);
}

template <typename T>
void divide_grad(const Tensor& x,
                 const Tensor& y,
                 const Tensor& out,
                 const Tensor& out_grad,
                 int axis,
                 Tensor* dx,
                 Tensor* dy) {
  if (dy) {
    // dy = -(x/y^2) * dout
    auto dy_res = -(x / y.pow(2.0)) * out_grad;
    if (x.dims() != y.dims()) {
      // Maybe need reduce here
      phi::DDim reduce_dim = get_reduce_dims(y.dims(), x.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dy_res, dy);
      } else {
        auto dy_reduce_res =
            dy_res.sum(phi::vectorize(reduce_dim), y.dtype(), false);
        auto dy_tmp = reshape<T>(dy_reduce_res, phi::vectorize(y.dims()));
        set_output<T>(dy_tmp, dy);
      }
    } else {
      set_output<T>(dy_res, dy);
    }
  }  // indicate we will compute dy
  if (dx) {
    // dx = (1/y) * dout
    auto one_tensor = full<T>(phi::vectorize(y.dims()), 1.0, y.dtype());
    auto dx_res = one_tensor / y * out_grad;
    if (y.dims() != x.dims()) {
      // Maybe need reduce here
      auto reduce_dim = get_reduce_dims(x.dims(), y.dims());
      if (!reduce_dim.size()) {
        set_output<T>(dx_res, dx);
      } else {
        auto dx_reduce_res =
            dx_res.sum(phi::vectorize(reduce_dim), x.dtype(), false);
        auto dx_tmp = reshape<T>(dx_reduce_res, phi::vectorize(x.dims()));
        set_output<T>(dx_tmp, dx);
      }

    } else {
      set_output<T>(dx_res, dx);
    }
  }  // indicate we will compute dx
}

template <typename T>
void sqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    auto x_grad_tmp = out_grad * 0.5 / out;
    set_output<T>(x_grad_tmp, x_grad);
  }
}

template <typename T>
void multiply_grad(const Tensor& x,
                   const Tensor& y,
                   const Tensor& out_grad,
                   int axis,
                   Tensor* x_grad,
                   Tensor* y_grad) {
  if (x_grad) {
    auto x_grad_unreduce = out_grad * y;
    if (x_grad_unreduce.dims() != x.dims()) {
      auto axes = get_reduce_dims_from_out(x_grad_unreduce.dims(), x.dims());
      if (!axes.size()) {
        set_output<T>(x_grad_unreduce, x_grad);
      } else {
        auto x_grad_reduced = x_grad_unreduce.sum(
            phi::vectorize(axes), x_grad_unreduce.dtype(), false);
        if (x_grad_reduced.dims().size() != x.dims().size()) {
          x_grad_reduced = reshape<T>(x_grad_reduced, x.shape());
        }
        set_output<T>(x_grad_reduced, x_grad);
      }
    } else {
      set_output<T>(x_grad_unreduce, x_grad);
    }
  }
  if (y_grad) {
    auto y_grad_unreduce = out_grad * x;
    if (y_grad_unreduce.dims() != y.dims()) {
      auto axes = get_reduce_dims_from_out(y_grad_unreduce.dims(), y.dims());
      if (!axes.size()) {
        set_output<T>(y_grad_unreduce, y_grad);
      } else {
        auto y_grad_reduced = y_grad_unreduce.sum(
            phi::vectorize(axes), y_grad_unreduce.dtype(), false);
        if (y_grad_reduced.dims().size() != y.dims().size()) {
          y_grad_reduced = reshape<T>(y_grad_reduced, y.shape());
        }
        set_output<T>(y_grad_reduced, y_grad);
      }
    } else {
      set_output<T>(y_grad_unreduce, y_grad);
    }
  }
}

template <typename T>
void expand_grad(const Tensor& x,
                 const Tensor& out_grad,
                 const IntArray& shape,
                 Tensor* x_grad) {
  if (x_grad) {
    auto out_dims = phi::make_ddim(shape.GetData());
    if (out_dims != x.dims()) {
      auto axes = get_reduce_dims(x.dims(), out_dims);
      if (!axes.size()) {
        by_pass<T>(out_grad, x_grad);
      } else {
        auto reduced = out_grad.sum(phi::vectorize(axes), x.dtype(), false);
        if (reduced.dims().size() != x.dims().size()) {
          reduced = reshape<T>(reduced, x.shape());
        }
        set_output<T>(reduced, x_grad);
      }
    } else {
      by_pass<T>(out_grad, x_grad);
    }
  }
}

template <typename T>
void exp_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {
  if (x_grad) {
    set_output<T>(out_grad * out, x_grad);
  }
}

template <typename T>
void slice_grad(const Tensor& input,
                const Tensor& out_grad,
                const std::vector<int64_t>& axes,
                const IntArray& starts,
                const IntArray& ends,
                const std::vector<int64_t>& infer_flags,
                const std::vector<int64_t>& decrease_axis,
                Tensor* input_grad) {
  if (input_grad) {
    size_t rank = input.dims().size();
    auto out_dims = out_grad.dims();
    auto in_dims = input.dims();

    auto decrease_size = decrease_axis.size();
    if (decrease_size > 0) {
      if (decrease_size == static_cast<size_t>(in_dims.size())) {
        // all dims decrease
        out_dims = phi::make_ddim(std::vector<int>(decrease_size, 1));
      } else {
        std::vector<int> origin_out_shape(out_dims.size() + decrease_size, -1);
        for (size_t i = 0; i < decrease_size; ++i) {
          origin_out_shape[decrease_axis[i]] = 1;
        }

        int index = 0;
        for (size_t i = 0; i < origin_out_shape.size(); ++i) {
          if (origin_out_shape[i] == -1) {
            origin_out_shape[i] = out_dims[index];
            ++index;
          }
        }
        out_dims = phi::make_ddim(origin_out_shape);
      }
    }

    std::vector<int> offsets(rank, 0);
    std::vector<int> extents(rank, 0);
    for (size_t i = 0; i < rank; ++i) {
      offsets[i] = 0;
      extents[i] = out_dims[i];
    }

    for (size_t i = 0; i < axes.size(); ++i) {
      int axis = axes[i];
      int64_t start = starts[i] < 0 ? (starts[i] + in_dims[axis]) : starts[i];
      start = std::max(start, static_cast<int64_t>(0));
      offsets[axis] = start;
    }

    std::vector<int> paddings;
    for (size_t i = 0; i < rank; ++i) {
      paddings.push_back(offsets[i]);
      paddings.push_back((in_dims[i] - out_dims[i]) - offsets[i]);
    }

    auto out_tmp = pad<T>(out_grad, paddings, 0.0);
    set_output<T>(out_tmp, input_grad);
  }
}

template <typename T>
void layer_norm_grad(const Tensor& x,
                     const paddle::optional<Tensor>& scale,
                     const paddle::optional<Tensor>& bias,
                     const Tensor& mean,
                     const Tensor& variance,
                     const Tensor& out_grad,
                     float epsilon,
                     int begin_norm_axis,
                     Tensor* x_grad,
                     Tensor* scale_grad,
                     Tensor* bias_grad) {
  auto x_dims = x.dims();
  auto shape_1 = 1;  // front part
  auto shape_2 = 1;  // back part
  for (int i = 0; i < begin_norm_axis; ++i) {
    shape_1 *= x_dims[i];
  }
  for (int i = begin_norm_axis; i < x.dims().size(); ++i) {
    shape_2 *= x_dims[i];
  }
  auto scale_ptr = scale.get_ptr();
  auto bias_ptr = bias.get_ptr();

  // cast dtype to float32 if dtype =float16
  Tensor x_cast = x;
  Tensor out_grad_cast = out_grad;
  Tensor scale_cast;
  if (scale_ptr) {
    scale_cast = reshape<T>(*scale_ptr, std::vector<int64_t>({1, shape_2}));
  }
  if (x.dtype() == phi::DataType::FLOAT16) {
    x_cast = cast<T>(x, phi::DataType::FLOAT32);
    out_grad_cast = cast<T>(out_grad, phi::DataType::FLOAT32);
    if (scale_ptr) {
      scale_cast = cast<T>(scale_cast, phi::DataType::FLOAT32);
    }
  }

  std::cout << "----------2----------" << std::endl;
  x_cast = reshape<T>(x_cast, std::vector<int64_t>({shape_1, shape_2}));
  std::cout << "----------3----------" << std::endl;
  auto out_grad_ =
      reshape<T>(out_grad_cast, std::vector<int64_t>({shape_1, shape_2}));
  std::cout << "----------4----------" << std::endl;
  auto mean_ = reshape<T>(mean, std::vector<int64_t>({shape_1, 1}));
  std::cout << "----------5----------" << std::endl;
  auto variance_ = reshape<T>(variance, std::vector<int64_t>({shape_1, 1}));
  std::cout << "----------6----------" << std::endl;
  if (bias_grad) {
    if (bias_ptr) {
      std::cout << "----------x----------" << std::endl;
      auto bias_grad_tmp =
          out_grad.sum(std::vector<int64_t>({0}), x.dtype(), false);
      std::cout << "----------y----------" << std::endl;
      set_output<T>(bias_grad_tmp, bias_grad);
    } else {
      std::cout << "----------z----------" << std::endl;
      bias_grad = nullptr;
    }
  }
  std::cout << "----------j----------" << std::endl;
  std::cout << x.dtype() << mean.dtype() << std::endl;
  std::cout << x_cast.dtype() << mean_.dtype() << std::endl;
  auto x_sub_mean = x_cast - mean_;
  auto sqrt_var_1 = sqrt<T>(1.0 / variance_);
  std::cout << "----------s----------" << std::endl;

  if (scale_grad) {
    if (scale_ptr) {
      std::cout << "----------r----------" << std::endl;
      auto scale_grad_tmp =
          (x_sub_mean * sqrt_var_1 * out_grad_cast)
              .sum(std::vector<int64_t>({0}), x.dtype(), false);
      std::cout << "----------n----------" << std::endl;
      set_output<T>(scale_grad_tmp, scale_grad);
    } else {
      std::cout << "----------q----------" << std::endl;
      scale_grad = nullptr;
    }
  }

  if (x_grad) {
    if (!scale_ptr) {
      scale_cast =
          full<T>(std::vector<int64_t>({1, shape_2}), 1.0, x_cast.dtype());
      std::cout << "----------scale_cast.type" << scale_cast.dtype()
                << std::endl;
    }
    std::cout << "--" << scale_cast.dtype() << sqrt_var_1.dtype()
              << out_grad_cast.dtype() << std::endl;
    auto dx_end = (scale_cast * sqrt_var_1 * out_grad_cast);
    std::cout << "----------b---------" << std::endl;
    auto d_mean_0 = (-sqrt_var_1 * out_grad_cast * scale_cast)
                        .sum(std::vector<int64_t>({1}), x_cast.dtype(), true);
    std::cout << "----------c----------" << std::endl;
    auto d_mean = 1.0 / shape_2 * d_mean_0;
    std::cout << "----------d----------" << std::endl;
    auto d_std_1 =
        (-(1.0 / variance_) * x_sub_mean * out_grad_cast * scale_cast)
            .sum(std::vector<int64_t>({1}), x_cast.dtype(), true);
    std::cout << "d_std_1.shape" << std::endl;
    std::cout << "d_std_1.shape" << d_std_1.dims() << std::endl;
    std::cout << "d_std_1.shape" << d_std_1.dims() << std::endl;
    auto d_std_2 = (1.0 / shape_2) * sqrt_var_1;
    std::cout << "----------7----------" << std::endl;
    d_std_2 = reshape<T>(d_std_2, std::vector<int64_t>({shape_1, 1}));
    std::cout << "----------8----------" << std::endl;
    d_std_2 = d_std_2 * x_sub_mean;
    std::cout << "----------9----------" << std::endl;
    auto d_std = d_std_1 * d_std_2;
    std::cout << "----------10----------" << std::endl;
    std::cout << "dx_end.shape" << dx_end.dims() << std::endl;
    std::cout << "d_mean.shape" << d_mean.dims() << std::endl;
    std::cout << "dx_std.shape" << d_std.dims() << std::endl;
    auto x_grad_tmp = dx_end + d_mean + d_std;
    std::cout << "----------11----------" << std::endl;
    x_grad_tmp = reshape<T>(x_grad_tmp, phi::vectorize(x.dims()));
    if (x.dtype() == phi::DataType::FLOAT16) {
      x_grad_tmp = cast<T>(x_grad_tmp, x.dtype());
    }
    set_output<T>(x_grad_tmp, x_grad);
  }
}
}  // namespace prim
}  // namespace paddle
