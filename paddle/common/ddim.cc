// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/ddim.h"

#include <set>

namespace common {

DDim::DDim() : rank_(-1) { dim_[0] = 0; }

DDim::DDim(const DDim& ddim) : dim_(), rank_(-1) { CopyFrom(ddim); }

DDim::DDim(const int* d, int n) : rank_(n) {
  dynamic_dim_assign(d, dim_.GetMutable(), n);
}

DDim::DDim(const int64_t* d, int n) : rank_(n) {
  dynamic_dim_assign(d, dim_.GetMutable(), n);
}

DDim::DDim(std::initializer_list<int64_t> init_list)
    : DDim(init_list.begin(), init_list.size()) {}

int64_t& DDim::at(int idx) {
  PADDLE_ENFORCE_GE(idx,
                    0,
                    common::errors::InvalidArgument(
                        "Invalid DDim index to be accessed. The valid index "
                        "is between 0 and %d, but received index is %d.",
                        rank_,
                        idx));
  PADDLE_ENFORCE_LT(idx,
                    rank_,
                    common::errors::InvalidArgument(
                        "Invalid DDim index to be accessed. The valid index "
                        "is between 0 and %d, but received index is %d.",
                        rank_,
                        idx));
  return dim_[idx];
}

int64_t DDim::at(int idx) const {
  PADDLE_ENFORCE_GE(idx,
                    0,
                    common::errors::InvalidArgument(
                        "Invalid DDim index to be accessed. The valid index "
                        "is between 0 and %d, but received index is %d.",
                        rank_,
                        idx));
  PADDLE_ENFORCE_LT(idx,
                    rank_,
                    common::errors::InvalidArgument(
                        "Invalid DDim index to be accessed. The valid index "
                        "is between 0 and %d, but received index is %d.",
                        rank_,
                        idx));
  return dim_[idx];
}

DDim make_ddim(std::initializer_list<int64_t> dims) {
  return DDim(dims.begin(), static_cast<int>(dims.size()));
}

DDim make_ddim(const std::vector<int64_t>& dims) {
  return DDim(dims.data(), static_cast<int>(dims.size()));
}

DDim make_ddim(const std::vector<int>& dims) {
  return DDim(dims.data(), static_cast<int>(dims.size()));
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
  if (size() == -1 && d.size() == -1) {
    return true;
  } else if (size() == -1 || d.size() == -1) {
    return false;
  } else {
    return size() == d.size() &&
           this->apply_visitor(DDimEqualityVisitor(d.Get()));
  }
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
  if (ddim.size() == -1) {
    return 0;
  }
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
      common::errors::InvalidArgument(
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
  if (ddim.size() == -1) {
    return os;
  }
  ddim.apply_visitor(DDimPrinter(os));
  return os;
}

DDim flatten_to_3d(const DDim& src, int num_row_dims, int num_col_dims) {
  PADDLE_ENFORCE_GE(src.size(),
                    3,
                    common::errors::InvalidArgument(
                        "The rank of src dim should be at least 3 "
                        "in flatten_to_3d, but received %d.",
                        src.size()));
  PADDLE_ENFORCE_EQ((num_row_dims >= 1 && num_row_dims < src.size()),
                    true,
                    common::errors::InvalidArgument(
                        "The num_row_dims should be inside [1, %d] "
                        "in flatten_to_3d, but received %d.",
                        src.size() - 1,
                        num_row_dims));
  PADDLE_ENFORCE_EQ((num_col_dims >= 2 && num_col_dims <= src.size()),
                    true,
                    common::errors::InvalidArgument(
                        "The num_col_dims should be inside [2, %d] "
                        "in flatten_to_3d, but received %d.",
                        src.size(),
                        num_col_dims));
  PADDLE_ENFORCE_GE(
      num_col_dims,
      num_row_dims,
      common::errors::InvalidArgument(
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
  if (ddim.size() > 0) strides[ddim.size() - 1] = 1;
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i + 1];
  }
  return strides;
}

DDim stride_numel(const DDim& ddim) {
  DDim strides;
  strides.rank_ = ddim.size();
  if (ddim.size() > 0) strides[ddim.size() - 1] = ddim[ddim.size() - 1];
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i];
  }
  return strides;
}

DDim DDim::reshape(std::vector<int>& shape) const {
  const DDim& in_dims = *this;

  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    if (shape[i] == 0) {
      shape[i] = static_cast<int>(in_dims.at(i));
    }
  }

  // Dim marked as "-1" must be inferred
  auto it = std::find(shape.begin(), shape.end(), -1);
  if (it != shape.end()) {
    int index = static_cast<int>(std::distance(shape.begin(), it));
    int reshape_out_product =
        std::accumulate(shape.begin(), shape.end(), -1, std::multiplies<>());
    shape[index] = static_cast<int>(product(in_dims)) / reshape_out_product;
  }

  return common::make_ddim(shape);
}

DDim DDim::transpose(const std::vector<int>& axis) const {
  const DDim& in_dims = *this;

  DDim out_dims(in_dims);
  for (int i = 0; i < static_cast<int>(axis.size()); i++) {
    out_dims[i] = in_dims[axis[i]];
  }
  return out_dims;
}

DDim ComputeCompatibleDim(const DDim& dim1, const DDim& dim2) {
  PADDLE_ENFORCE_EQ(dim1.size() == dim2.size(),
                    true,
                    "Does not support rank inconsistency: rank1=%d, rank2=%d",
                    dim1.size(),
                    dim2.size());
  std::vector<int64_t> result;
  for (int i = 0; i < dim1.size(); ++i) {
    if (dim1[i] != dim2[i]) {
      result.push_back(-1);
    } else {
      result.push_back(dim1[i]);
    }
  }
  return make_ddim(result);
}

bool AreDimsWithDynamicShapeCompatible(const DDim& dim1, const DDim& dim2) {
  if (dim1.size() != dim2.size()) {
    return false;
  }
  for (int i = 0; i < dim1.size(); ++i) {
    if (dim1[i] >= 0 && dim2[i] >= 0 && dim1[i] != dim2[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace common

namespace std {

std::size_t hash<common::DDim>::operator()(common::DDim const& ddim) const {
  int ndim = ddim.size();
  std::size_t seed = ndim;
  for (int i = 0; i < ndim; ++i) {
    seed ^= ddim.Get()[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

}  // namespace std
