/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "glog/logging.h"
#include "paddle/common/macros.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi::funcs {

class TensorAssign {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

class ReduceAdd {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data += *src_data;
  }
};
static ReduceAdd reduce_add;

class ReduceMultiply {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data *= *src_data;
  }
};
static ReduceMultiply reduce_mul;

class ReduceMax {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data > *self_data ? *src_data : *self_data;
  }
};
static ReduceMax reduce_max;

class ReduceMin {
 public:
  template <typename tensor_t>
  void operator()(tensor_t* self_data, tensor_t* src_data) const {
    *self_data = *src_data < *self_data ? *src_data : *self_data;
  }
};
static ReduceMin reduce_min;

template <typename T>
inline T IntFloorDiv(T a, T b) {
  if ((a < 0) != (b < 0)) {
    // compute div and mod at the same time can be optimized by compilers
    const auto quot = a / b;
    const auto rem = a % b;
    return rem ? quot - 1 : quot;
  }
  return a / b;
}

/**
 * A divmod free solution for faster offset mapping. This class only do the
 * necessary multiplication, therefore the computation and memory access should
 * be lower than divmod and naive index mapping. Usage:
 *
 * \code
 * CoordinateManager<true> cm(index_shape, self_strides, ndim,
 * axis_to_put, &src_strides);
 *
 * for (int i = 0; i < index_shape.numel(); i++) {
 *    index_t index = index_data[i];
 *    cm.CalculateOffset(index_t);
 *    int64_t replace_self_index = cm.offset1;
 *    int64_t replace_src_index = cm.offset2;
 *    ...
 * }
 * \endcode
 */
template <bool compute_both = false>
class CoordinateManager {
 private:
  const phi::DDim& shape;
  const phi::DDim& strides1;
  const int ndim;
  const int src_dim;
  int64_t last_offset;
  std::vector<int64_t> indices;
  const phi::DDim* strides2;

 public:
  int64_t offset1;
  int64_t offset2;

  CoordinateManager(const phi::DDim& _shape,
                    const phi::DDim& _strides1,
                    int _ndim,
                    int _src_dim,
                    const phi::DDim* _strides2 = nullptr)
      : shape(_shape),
        strides1(_strides1),
        ndim(_ndim),
        src_dim(_src_dim),
        last_offset(0),
        strides2(_strides2),
        offset1(0),
        offset2(0) {
    indices.resize(ndim, 0);
    // calculate correct starting offsets
    if (ndim - 1 != _src_dim) offset1 = -strides1[ndim - 1];
    if constexpr (compute_both) offset2 = -strides2->operator[](ndim - 1);
  }

  template <typename index_t>
  void CalculateOffset(index_t index) {
    int change_dim = ndim - 1;
    // step 1: calculate the carry or borrow dim
    for (int dim = ndim - 1; dim > 0; dim--) {
      if (indices[dim] >= shape[dim]) {
        indices[dim] = 0;
        change_dim = dim - 1;
        // carry or borrow operation: we do not check boundaries here, please
        // make sure that do not call map_offset more than index.numel(),
        // otherwise we will have illegal access
        ++indices[change_dim];
      }
    }

    // step 2: update the axis to put/take offset
    offset1 -= last_offset;
    last_offset = index * strides1[src_dim];
    offset1 += last_offset;

    // step 3: clear the offset due to carry using minimum number of `mul`s.
    // skip all src_dim related computation, since they have independent
    // logics. Also, if strides2 (compute both) is available, compute the
    // offset (usually for src tensor).

    if (change_dim != src_dim) offset1 += strides1[change_dim];
    if constexpr (compute_both) offset2 += strides2->operator[](change_dim);
    for (int dim = change_dim + 1; dim < ndim; dim++) {
      int dim_max_index = shape[dim] - 1;
      // clear the tail elements after the carrying dim
      if constexpr (compute_both)
        offset2 -= strides2->operator[](dim) * dim_max_index;
      if (dim == src_dim) continue;
      offset1 -= strides1[dim] * dim_max_index;
    }
    ++indices.back();
  }
};

/**
 * Used in some of the value grad calculation, since those compute indices in a
 * back-to-front order. Decide not to fuse with CoordinateManager via
 * templating, otherwise the readability will be bad.
 */
template <bool compute_both = false>
class ReversedCoordinateManager {
 private:
  const phi::DDim& shape;
  const phi::DDim& strides1;
  const int ndim;
  const int src_dim;
  int64_t last_offset;
  std::vector<int64_t> indices;
  const phi::DDim* strides2;

 public:
  int64_t offset1;
  int64_t offset2;

  ReversedCoordinateManager(const phi::DDim& _shape,
                            const phi::DDim& _strides1,
                            int _ndim,
                            int _src_dim,
                            const phi::DDim* _strides2 = nullptr)
      : shape(_shape),
        strides1(_strides1),
        ndim(_ndim),
        src_dim(_src_dim),
        last_offset(0),
        strides2(_strides2),
        offset1(0),
        offset2(0) {
    indices.resize(ndim, 0);
    // reversed should have an extra stride.back()
    if (ndim - 1 != _src_dim) offset1 = strides1[ndim - 1];
    if constexpr (compute_both) offset2 = strides2->operator[](ndim - 1);
    for (int i = 0; i < _ndim; i++) {
      indices[i] = _shape[i] - 1;
      if constexpr (compute_both)
        offset2 += strides2->operator[](i) * indices[i];
      if (i == src_dim) continue;
      offset1 += strides1[i] * indices[i];
    }
  }

  template <typename index_t>
  void CalculateOffset(index_t index) {
    int change_dim = ndim - 1;
    // step 1: calculate the borrow dim
    for (int dim = ndim - 1; dim > 0; dim--) {
      if (indices[dim] < 0) {
        indices[dim] = shape[dim] - 1;
        change_dim = dim - 1;
        --indices[change_dim];
      }
    }

    // step 2: update the axis to put/take offset
    offset1 -= last_offset;
    last_offset = index * strides1[src_dim];
    offset1 += last_offset;

    // step 3: clear the offset due to borrow using minimum number of `mul`s.

    if (change_dim != src_dim) offset1 -= strides1[change_dim];
    if constexpr (compute_both) offset2 -= strides2->operator[](change_dim);
    for (int dim = change_dim + 1; dim < ndim; dim++) {
      int dim_max_index = shape[dim] - 1;
      // clear the tail elements after the carrying dim
      if constexpr (compute_both)
        offset2 += strides2->operator[](dim) * dim_max_index;
      if (dim == src_dim) continue;
      offset1 += strides1[dim] * dim_max_index;
    }
    --indices.back();
  }
};

template <typename tensor_t,
          typename index_t = int64_t,
          bool is_scatter_like = true>
struct cpu_gather_scatter_functor {
  template <typename func_t>
  void operator()(phi::DenseTensor self,
                  int dim,
                  const phi::DenseTensor& index,
                  const phi::DenseTensor& src,
                  const std::string& method_name,
                  const func_t& reduce_op,
                  bool include_self,
                  const phi::DeviceContext& dev_ctx UNUSED) {
    if (index.numel() == 0) {
      return;
    }
    auto* self_data = self.data<tensor_t>();
    auto* index_data = index.data<index_t>();
    auto* src_data = src.data<tensor_t>();
    int64_t self_size = self.numel();
    int64_t index_size = index.numel();
    int64_t src_size = src.numel();
    auto self_dims = self.dims();
    auto src_dims = src.dims();

    const bool is_gather_or_scatter_assign =
        method_name == "gather" || method_name == "assign";

    if (self_size == 0 || src_size == 0 || index_size == 0) {
      VLOG(3) << "zero size input found";
      common::errors::InvalidArgument(
          "self_size, src_size, index_size cannot be 0");
      return;
    }
    int self_select_dim_size = self_dims[dim];
    int src_select_dim_size = src_dims[dim];

    // gather and assign do not need nums_of_elements
    std::vector<int> nums_of_elements;
    if (!is_gather_or_scatter_assign) nums_of_elements.resize(self.numel(), 0);

    const int ndim = index.dims().size();

    CoordinateManager<is_scatter_like> cm(
        index.dims(),
        is_scatter_like ? self.strides() : src.strides(),
        ndim,
        dim,
        &src.strides());

    for (int64_t i = 0; i < index_size; i++) {
      int64_t index = index_data[i];

      int64_t replace_index_self = 0, replace_index_src = 0;
      // offset1 is always related to index
      if constexpr (is_scatter_like) {
        PADDLE_ENFORCE_EQ(
            (index >= -self_select_dim_size) && (index < self_select_dim_size),
            true,
            common::errors::OutOfRange(
                "Variable value (index) of scatter cpu kernel, "
                "expected >= %d and < %d, but got %ld."
                "Please check the input value.",
                -self_select_dim_size,
                self_select_dim_size,
                index));
        if (index < 0) index += self_select_dim_size;
        cm.CalculateOffset(index);
        replace_index_self = cm.offset1;
        replace_index_src = cm.offset2;
      } else {
        PADDLE_ENFORCE_EQ(
            (index >= -src_select_dim_size) && (index < src_select_dim_size),
            true,
            common::errors::OutOfRange(
                "Variable value (index) of gather cpu kernel, "
                "expected >= %d and < %d, but got %ld."
                "Please check the input value.",
                -src_select_dim_size,
                src_select_dim_size,
                index));
        if (index < 0) index += src_select_dim_size;
        cm.CalculateOffset(index);
        replace_index_self = i;
        replace_index_src = cm.offset1;
      }

      if (include_self == false && is_gather_or_scatter_assign == false &&
          nums_of_elements[replace_index_self] == 0) {
        self_data[replace_index_self] = src_data[replace_index_src];
      } else {
        reduce_op((tensor_t*)(self_data + replace_index_self),  // NOLINT
                  (tensor_t*)(src_data + replace_index_src));   // NOLINT
      }
      if (!is_gather_or_scatter_assign)
        nums_of_elements[replace_index_self] += 1;
    }

    if (method_name == "mean") {
      if (include_self) {
        for (int i = 0; i < self_size; i++) {
          if (!nums_of_elements[i]) continue;
          if constexpr (std::is_integral_v<std::decay_t<tensor_t>>) {
            self_data[i] = IntFloorDiv(
                self_data[i], static_cast<tensor_t>(nums_of_elements[i] + 1));
          } else {
            self_data[i] =
                self_data[i] / static_cast<tensor_t>(nums_of_elements[i] + 1);
          }
        }
      } else {
        for (int i = 0; i < self_size; i++) {
          if (!nums_of_elements[i]) continue;
          if constexpr (std::is_integral_v<std::decay_t<tensor_t>>) {
            self_data[i] = IntFloorDiv(
                self_data[i], static_cast<tensor_t>(nums_of_elements[i]));
          } else {
            self_data[i] =
                self_data[i] / static_cast<tensor_t>(nums_of_elements[i]);
          }
        }
      }
    }
  }
};

template <typename tensor_t, typename index_t>
void cpu_gather_kernel(phi::DenseTensor self,
                       int dim,
                       const phi::DenseTensor& index,
                       phi::DenseTensor result,
                       bool include_self,
                       const phi::DeviceContext& dev_ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/false>()(
      result, dim, index, self, "gather", tensor_assign, include_self, dev_ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_assign_kernel(phi::DenseTensor self,
                               int dim,
                               const phi::DenseTensor& index,
                               phi::DenseTensor src,
                               bool include_self,
                               const phi::DeviceContext& dev_ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "assign", tensor_assign, include_self, dev_ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_add_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& dev_ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "add", reduce_add, include_self, dev_ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& dev_ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "mul", reduce_mul, include_self, dev_ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mean_kernel(phi::DenseTensor self,
                             int dim,
                             const phi::DenseTensor& index,
                             phi::DenseTensor src,
                             bool include_self,
                             const phi::DeviceContext& dev_ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "mean", reduce_add, include_self, dev_ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_max_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& dev_ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "max", reduce_max, include_self, dev_ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_min_kernel(phi::DenseTensor self,
                            int dim,
                            const phi::DenseTensor& index,
                            phi::DenseTensor src,
                            bool include_self,
                            const phi::DeviceContext& dev_ctx) {
  cpu_gather_scatter_functor<tensor_t,
                             index_t,
                             /*is_scatter_like=*/true>()(
      self, dim, index, src, "min", reduce_min, include_self, dev_ctx);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_input_grad_kernel(phi::DenseTensor self UNUSED,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self UNUSED,
                                   const phi::DeviceContext& dev_ctx UNUSED) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  const int ndim = index.dims().size();
  const int64_t index_size = index.numel();
  CoordinateManager<false> cm(index.dims(), grad.strides(), ndim, dim, nullptr);

  for (int64_t i = 0; i < index_size; i++) {
    int64_t index = index_data[i];
    cm.CalculateOffset(index);
    int64_t replace_index = cm.offset1;
    grad_data[replace_index] = 0;
  }
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_min_max_input_grad_kernel(
    phi::DenseTensor self UNUSED,
    int dim,
    const phi::DenseTensor& index,
    const phi::DenseTensor& out,
    const phi::DenseTensor& x,
    const phi::DenseTensor& value,
    phi::DenseTensor grad,
    const std::string& reduce,
    bool include_self UNUSED,
    const phi::DeviceContext& dev_ctx) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();
  auto* out_data = out.data<tensor_t>();
  auto* x_data = x.data<tensor_t>();
  auto* value_data = value.data<tensor_t>();

  const int ndim = index.dims().size();
  const int64_t index_size = index.numel();
  const int64_t grad_size = grad.numel();
  // only amin/amax needs the offset2, but we compute together anyway.
  CoordinateManager<true> cm(
      index.dims(), grad.strides(), ndim, dim, &value.strides());

  // make sure that reduce in {'mul', 'multiply', 'amin', 'amax'}
  const bool is_mul = reduce == "multiply" || reduce == "mul";
  std::vector<int> num_elements(grad.numel(), 0);
  for (int64_t i = 0; i < index_size; i++) {
    int64_t index = index_data[i];
    cm.CalculateOffset(index);
    int64_t replace_index_grad = cm.offset1;
    if (is_mul && num_elements[replace_index_grad] == 0) {
      grad_data[replace_index_grad] = static_cast<tensor_t>(
          grad_data[replace_index_grad] * out_data[replace_index_grad] /
          x_data[replace_index_grad]);
      num_elements[replace_index_grad] += 1;
    } else if (!is_mul) {
      if (out_data[replace_index_grad] != x_data[replace_index_grad]) {
        grad_data[replace_index_grad] = 0;
      } else {
        int64_t replace_index_value = cm.offset2;
        if (out_data[replace_index_grad] == value_data[replace_index_value])
          num_elements[replace_index_grad] += 1;
      }
    }
  }

  // TODO(heqianyue): I don't think the origin impl is correct, what about
  // include_self = False?
  if (!is_mul) {
    for (int64_t i = 0; i < grad_size; i++) {
      grad_data[i] = grad_data[i] / static_cast<tensor_t>(num_elements[i] + 1);
    }
  }
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mean_input_grad_kernel(phi::DenseTensor self UNUSED,
                                        int dim,
                                        const phi::DenseTensor& index,
                                        phi::DenseTensor grad,
                                        bool include_self UNUSED,
                                        const phi::DeviceContext& dev_ctx
                                            UNUSED) {
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  int64_t grad_size = grad.numel();

  const int ndim = index.dims().size();
  const int64_t index_size = index.numel();
  CoordinateManager<false> cm(index.dims(), grad.strides(), ndim, dim, nullptr);
  std::vector<int> num_elements(grad_size, 0);
  for (int64_t i = 0; i < index_size; i++) {
    int64_t index = index_data[i];
    cm.CalculateOffset(index);
    int64_t replace_index = cm.offset1;
    num_elements[replace_index] += 1;
  }
  for (int64_t i = 0; i < grad_size; i++)
    if (num_elements[i])
      grad_data[i] = grad_data[i] / static_cast<tensor_t>(num_elements[i] + 1);
}

template <typename tensor_t, typename index_t>
void cpu_scatter_value_grad_kernel(phi::DenseTensor self,
                                   int dim,
                                   const phi::DenseTensor& index,
                                   phi::DenseTensor grad,
                                   bool include_self UNUSED,
                                   const phi::DeviceContext& dev_ctx UNUSED) {
  const auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  std::vector<bool> is_self_grad_used(self.numel(), false);

  const int ndim = index.dims().size();
  ReversedCoordinateManager<true> cm(
      index.dims(), self.strides(), ndim, dim, &grad.strides());

  for (int64_t i = index.numel() - 1; i >= 0; i--) {
    int64_t index = index_data[i];
    cm.CalculateOffset(index);
    int64_t replace_index_self = cm.offset1;
    int64_t replace_index_grad = cm.offset2;
    if (!is_self_grad_used[replace_index_self]) {
      grad_data[replace_index_grad] = self_data[replace_index_self];
      is_self_grad_used[replace_index_self] = true;
    }
  }
}

template <typename tensor_t, typename index_t>
void cpu_scatter_add_mean_value_grad_kernel(phi::DenseTensor self,
                                            int dim,
                                            const phi::DenseTensor& index,
                                            const phi::DenseTensor& out UNUSED,
                                            const phi::DenseTensor& x UNUSED,
                                            const phi::DenseTensor& value
                                                UNUSED,
                                            phi::DenseTensor grad,
                                            const std::string& reduce,
                                            bool include_self,
                                            const phi::DeviceContext& dev_ctx) {
  const auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();

  int64_t self_size = self.numel();

  phi::funcs::set_constant(dev_ctx, &grad, 0);

  std::vector<int> num_elements;
  const int ndim = index.dims().size();

  // Note: make sure that `reduce` in {'mean', 'add'}.
  const bool is_mean = reduce == "mean";
  if (is_mean) {
    num_elements.resize(self_size, static_cast<int>(include_self));
    ReversedCoordinateManager<false> cm(
        index.dims(), self.strides(), ndim, dim, nullptr);

    for (int64_t i = index.numel() - 1; i >= 0; i--) {
      int64_t index = index_data[i];
      cm.CalculateOffset(index);
      int64_t replace_index_self = cm.offset1;
      num_elements[replace_index_self] += 1;
    }
  }

  ReversedCoordinateManager<true> cm(
      index.dims(), self.strides(), ndim, dim, &grad.strides());
  for (int64_t i = index.numel() - 1; i >= 0; i--) {
    int64_t index = index_data[i];
    cm.CalculateOffset(index);
    int64_t replace_index_self = cm.offset1;
    int64_t replace_index_grad = cm.offset2;
    if (is_mean) {
      grad_data[replace_index_grad] =
          self_data[replace_index_self] /
          static_cast<tensor_t>(num_elements[replace_index_self]);
    } else {
      grad_data[replace_index_grad] = self_data[replace_index_self];
    }
  }
}

template <typename tensor_t, typename index_t>
void cpu_scatter_mul_min_max_value_grad_kernel(
    phi::DenseTensor self,
    int dim,
    const phi::DenseTensor& index,
    const phi::DenseTensor& out,
    const phi::DenseTensor& x,
    const phi::DenseTensor& value,
    phi::DenseTensor grad,
    const std::string& reduce,
    bool include_self,
    const phi::DeviceContext& dev_ctx) {
  const auto* self_data = self.data<tensor_t>();
  auto* index_data = index.data<index_t>();
  auto* grad_data = grad.data<tensor_t>();
  auto* out_data = out.data<tensor_t>();
  auto* x_data = x.data<tensor_t>();
  auto* value_data = value.data<tensor_t>();

  std::vector<int> num_elements;
  const bool is_min_max = reduce == "amin" || reduce == "amax";
  if (is_min_max) num_elements.resize(self.numel(), 0);

  const int ndim = index.dims().size();
  const int64_t index_size = index.numel();
  {  // `cm` should be destroyed once the computation is done, no reuse
    CoordinateManager<true> cm(
        index.dims(), self.strides(), ndim, dim, &grad.strides());
    for (int64_t i = 0; i < index_size; i++) {
      int64_t index = index_data[i];
      cm.CalculateOffset(index);
      int64_t replace_index_self = cm.offset1;
      int64_t replace_index_grad = cm.offset2;
      if (is_min_max &&
          out_data[replace_index_self] == value_data[replace_index_grad]) {
        num_elements[replace_index_self] += 1;
      } else if (!is_min_max) {
        grad_data[replace_index_grad] =
            self_data[replace_index_self] *
            (out_data[replace_index_self] / value_data[replace_index_grad]);
      }
    }
  }

  if (is_min_max) {
    CoordinateManager<true> cm(
        index.dims(), self.strides(), ndim, dim, &grad.strides());
    for (int64_t i = 0; i < index_size; i++) {
      int64_t index = index_data[i];
      cm.CalculateOffset(index);
      int64_t replace_index_self = cm.offset1;
      int64_t replace_index_grad = cm.offset2;
      if (out_data[replace_index_self] == value_data[replace_index_grad]) {
        if (out_data[replace_index_self] == x_data[replace_index_self])
          grad_data[replace_index_grad] =
              self_data[replace_index_self] /
              static_cast<tensor_t>(num_elements[replace_index_self] + 1);
        else
          grad_data[replace_index_grad] =
              self_data[replace_index_self] /
              static_cast<tensor_t>(num_elements[replace_index_self]);
      }
    }
  }
}

Instantiate_Template_Function(cpu_gather_kernel)                  // NOLINT
    Instantiate_Template_Function(cpu_scatter_assign_kernel)      // NOLINT
    Instantiate_Template_Function(cpu_scatter_add_kernel)         // NOLINT
    Instantiate_Template_Function(cpu_scatter_mul_kernel)         // NOLINT
    Instantiate_Template_Function(cpu_scatter_mean_kernel)        // NOLINT
    Instantiate_Template_Function(cpu_scatter_max_kernel)         // NOLINT
    Instantiate_Template_Function(cpu_scatter_min_kernel)         // NOLINT
    Instantiate_Template_Function(cpu_scatter_input_grad_kernel)  // NOLINT
    Instantiate_Template_Function(cpu_scatter_value_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        cpu_scatter_mul_min_max_input_grad_kernel)                     // NOLINT
    Instantiate_Template_Function(cpu_scatter_mean_input_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        cpu_scatter_add_mean_value_grad_kernel)  // NOLINT
    Instantiate_Template_Function_With_Out(
        cpu_scatter_mul_min_max_value_grad_kernel)  // NOLINT

}  // namespace phi::funcs
