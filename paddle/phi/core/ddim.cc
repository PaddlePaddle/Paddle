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

#include "paddle/phi/core/ddim.h"

#include <set>

namespace phi {

DDim make_ddim(std::initializer_list<int64_t> dims) {
  return DDim(dims.begin(), dims.size());
}

DDim make_ddim(const std::vector<int64_t>& dims) {
  return DDim(dims.data(), dims.size());
}

DDim make_ddim(const std::vector<int>& dims) {
  return DDim(dims.data(), dims.size());
}

struct DDimEqualityVisitor {
  explicit DDimEqualityVisitor(const int64_t* d) : d_(d) {}

  template <int D>
  inline bool operator()(const Dim<D>& self) const {
    return UnrollCompare<D>::Run(self.Get(), d_);
  }

  const int64_t* d_;
};

bool DDim::operator==(const DDim& d) const {
  return size() == d.size() &&
         this->apply_visitor(DDimEqualityVisitor(d.Get()));
}

bool DDim::operator!=(const DDim& d) const { return !(*this == d); }

std::string DDim::to_str() const {
  std::stringstream ss;
  ss << '[';
  if (rank_ > 0) ss << dim_[0];

  for (int i = 1; i < rank_; ++i) ss << ", " << dim_[i];
  ss << ']';
  return ss.str();
}

struct ProductVisitor {
  template <int D>
  inline int64_t operator()(const Dim<D>& dim) {
    return product(dim);
  }
};

int64_t product(const DDim& ddim) {
  return ddim.apply_visitor(ProductVisitor());
}

bool contain_unknown_dim(const DDim& ddim) {
  for (int i = 0; i < ddim.size(); ++i) {
    if (ddim[i] < 0) {
      return true;
    }
  }

  return false;
}

DDim slice_ddim(const DDim& dim, int begin, int end) {
  PADDLE_ENFORCE_EQ(
      (begin >= 0 && end <= dim.size()),
      true,
      phi::errors::InvalidArgument(
          "[begin(%d), end(%d)) must be inside [0, %d) in ddim slice.",
          begin,
          end,
          dim.size()));
  // Constructor of DDim would check whether end - begin is valid
  return DDim(dim.Get() + begin, end - begin);
}

int arity(const DDim& d) { return d.size(); }

struct DDimPrinter {
  std::ostream& os;
  explicit DDimPrinter(std::ostream& os_) : os(os_) {}

  template <int D>
  void operator()(const Dim<D>& t) {
    os << t;
  }
};

std::ostream& operator<<(std::ostream& os, const DDim& ddim) {
  ddim.apply_visitor(DDimPrinter(os));
  return os;
}

DDim flatten_to_3d(const DDim& src, int num_row_dims, int num_col_dims) {
  PADDLE_ENFORCE_GE(
      src.size(),
      3,
      phi::errors::InvalidArgument("The rank of src dim should be at least 3 "
                                   "in flatten_to_3d, but received %d.",
                                   src.size()));
  PADDLE_ENFORCE_EQ(
      (num_row_dims >= 1 && num_row_dims < src.size()),
      true,
      phi::errors::InvalidArgument("The num_row_dims should be inside [1, %d] "
                                   "in flatten_to_3d, but received %d.",
                                   src.size() - 1,
                                   num_row_dims));
  PADDLE_ENFORCE_EQ(
      (num_col_dims >= 2 && num_col_dims <= src.size()),
      true,
      phi::errors::InvalidArgument("The num_col_dims should be inside [2, %d] "
                                   "in flatten_to_3d, but received %d.",
                                   src.size(),
                                   num_col_dims));
  PADDLE_ENFORCE_GE(
      num_col_dims,
      num_row_dims,
      phi::errors::InvalidArgument(
          "The num_row_dims should be less than num_col_dims in flatten_to_3d,"
          "but received num_row_dims = %d, num_col_dims = %d.",
          num_row_dims,
          num_col_dims));

  return DDim({product(slice_ddim(src, 0, num_row_dims)),
               product(slice_ddim(src, num_row_dims, num_col_dims)),
               product(slice_ddim(src, num_col_dims, src.size()))});
}

DDim flatten_to_2d(const DDim& src, int num_col_dims) {
  return DDim({product(slice_ddim(src, 0, num_col_dims)),
               product(slice_ddim(src, num_col_dims, src.size()))});
}

DDim flatten_to_1d(const DDim& src) { return DDim({product(src)}); }

DDim stride(const DDim& ddim) {
  DDim strides;
  strides.rank_ = ddim.size();
  strides[ddim.size() - 1] = 1;
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i + 1];
  }
  return strides;
}

DDim stride_numel(const DDim& ddim) {
  DDim strides;
  strides.rank_ = ddim.size();
  strides[ddim.size() - 1] = ddim[ddim.size() - 1];
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i];
  }
  return strides;
}

DDim DDim::reshape(const std::vector<int>& shape) const {
  const int64_t copy_dim_val = 0;
  const DDim& in_dims = *this;
  DDim out_dims;
  out_dims.rank_ = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == copy_dim_val) {
      PADDLE_ENFORCE_LT(static_cast<int>(i),
                        in_dims.size(),
                        phi::errors::InvalidArgument(
                            "Index %d of shape under which the value of 0 "
                            "is stored, must be lower than the number of "
                            "old dimensions. But received shape[%d] = 0, "
                            "dimensions = %d, shape = [%s].",
                            i,
                            in_dims.size(),
                            in_dims));
      out_dims[i] = in_dims[i];
    } else {
      out_dims[i] = shape[i];
    }
  }
  return out_dims;
}

DDim DDim::transpose(const std::vector<int>& axis) const {
  const DDim& in_dims = *this;
  size_t in_rank = in_dims.size();
  size_t axis_size = axis.size();

  auto axis_set = std::set<int>(axis.begin(), axis.end());
  PADDLE_ENFORCE_EQ(axis_set.size(),
                    axis_size,
                    phi::errors::InvalidArgument(
                        "In an axis array, elements must be unique."));

  PADDLE_ENFORCE_EQ(
      in_rank,
      axis_size,
      phi::errors::InvalidArgument("The input dimension's size "
                                   "should be equal to the axis's size. "
                                   "But received dimension is %d, "
                                   "axis's size is %d",
                                   in_rank,
                                   axis_size));

  PADDLE_ENFORCE_LT(*std::max_element(axis.begin(), axis.end()),
                    axis_size,
                    phi::errors::InvalidArgument(
                        "Axis values must be ranging from 0 to (dims - 1)."));

  DDim out_dims(in_dims);
  for (size_t i = 0; i < axis_size; i++) {
    out_dims[i] = in_dims[axis[i]];
  }
  return out_dims;
}

}  // namespace phi
