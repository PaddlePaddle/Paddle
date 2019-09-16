/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

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
  PADDLE_ENFORCE(begin >= 0 && end <= dim.size(),
                 "[begin(%d), end(%d)) must be inside [0, %d) in ddim slice.",
                 begin, end, dim.size());
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

}  // namespace framework
}  // namespace paddle
