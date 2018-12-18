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

template <typename T>
struct DDimAssignFunctor {
  static_assert(std::is_integral<T>::value, "T must be integral type");
  using result_type = void;
  explicit DDimAssignFunctor(const T* in) : in_(in) {}

  template <int D>
  inline void operator()(Dim<D>& dim) {  // NOLINT
    UnrollAssign<D>::Run(in_, dim.data());
  }

  const T* in_;
};

DDim::DDim(const int* d, int n) : rank_(n) {
  this->apply_visitor(DDimAssignFunctor<int>(d));
}

DDim::DDim(const int64_t* d, int n) : rank_(n) {
  this->apply_visitor(DDimAssignFunctor<int64_t>(d));
}

template <int N>
Dim<N> make_dim(const int64_t* d) {
  Dim<N> ret;
  for (int i = 0; i < N; ++i) ret[i] = d[i];
  return ret;
}

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
    return UnrollCompare<D>::Run(self.data(), d_);
  }

  const int64_t* d_;
};

bool DDim::operator==(const DDim& d) const {
  return rank_ == d.rank_ && this->apply_visitor(DDimEqualityVisitor(d.data()));
}

bool DDim::operator!=(const DDim& d) const { return !(*this == d); }

struct DDimPlusVisitor {
  explicit DDimPlusVisitor(const int64_t* d1, const int64_t* d2)
      : d1_(d1), d2_(d2) {}

  template <int D>
  inline void operator()(Dim<D>& self) const {
    UnrollAdd<D>::Run(d1_, d2_, self.data());
  }

  const int64_t* d1_;
  const int64_t* d2_;
};

DDim DDim::operator+(const DDim& d) const {
  PADDLE_ENFORCE(rank_ == d.rank_);
  DDim ret;
  ret.rank_ = rank_;
  ret.apply_visitor(DDimPlusVisitor(data(), d.data()));
  return ret;
}

struct DDimMulVisitor {
  explicit DDimMulVisitor(const int64_t* d1, const int64_t* d2)
      : d1_(d1), d2_(d2) {}

  template <int D>
  inline void operator()(Dim<D>& self) const {
    UnrollMul<D>::Run(d1_, d2_, self.data());
  }

  const int64_t* d1_;
  const int64_t* d2_;
};

DDim DDim::operator*(const DDim& d) const {
  PADDLE_ENFORCE(rank_ == d.rank_);
  DDim ret;
  ret.rank_ = rank_;
  ret.apply_visitor(DDimMulVisitor(data(), d.data()));
  return ret;
}

int64_t get(const DDim& ddim, int idx) { return ddim[idx]; }

void set(DDim& ddim, int idx, int value) { ddim[idx] = value; }  // NOLINT

std::vector<int64_t> vectorize(const DDim& ddim) {
  std::vector<int64_t> result(DDim::kMaxRank);
  for (int i = 0; i < ddim.size(); ++i) {
    result[i] = ddim[i];
  }
  result.resize(ddim.size());
  return result;
}

// NOTE: framework::vectorize converts to type int64_t
//       which does not fit cudnn inputs.
std::vector<int> vectorize2int(const DDim& ddim) {
  std::vector<int> result(DDim::kMaxRank);
  for (int i = 0; i < ddim.size(); ++i) {
    result[i] = ddim[i];
  }
  result.resize(ddim.size());
  return result;
}

struct ProductVisitor {
  template <int D>
  int64_t operator()(const Dim<D>& dim) {
    return product(dim);
  }
};

int64_t product(const DDim& ddim) {
  return ddim.apply_visitor(ProductVisitor());
}

DDim slice_ddim(const DDim& dim, int begin, int end) {
  PADDLE_ENFORCE(begin < end,
                 "Begin index must be less than end index in ddim slice.");
  PADDLE_ENFORCE(begin >= 0,
                 "Begin index can't be less than zero in ddim slice.");
  DDim ret;
  ret.rank_ = end - begin;
  for (int i = 0; i < ret.rank_; ++i) {
    ret[i] = dim[i + begin];
  }
  return ret;
}

int arity(const DDim& d) { return d.size(); }

/// \cond HIDDEN

struct DDimPrinter {
  std::ostream& os;
  explicit DDimPrinter(std::ostream& os_) : os(os_) {}

  template <typename T>
  void operator()(const T& t) {
    os << t;
  }
};

/// \endcond

std::ostream& operator<<(std::ostream& os, const DDim& ddim) {
  ddim.apply_visitor(DDimPrinter(os));
  return os;
}

DDim flatten_to_2d(const DDim& src, int num_col_dims) {
  int rank = src.size();
  return make_ddim({product(slice_ddim(src, 0, num_col_dims)),
                    product(slice_ddim(src, num_col_dims, rank))});
}

DDim flatten_to_1d(const DDim& src) { return make_ddim({product(src)}); }

DDim stride(const DDim& ddim) {
  DDim strides;
  strides.rank_ = ddim.size();
  strides[ddim.size() - 1] = 1;
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i + 1];
  }
  return strides;
}

DDim stride_numel(const framework::DDim& ddim) {
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
