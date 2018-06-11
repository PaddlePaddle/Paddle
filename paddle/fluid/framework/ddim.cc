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

/// @cond HIDDEN

template <int i>
Dim<i> make_dim(const int64_t* d) {
  return Dim<i>(*d, make_dim<i - 1>(d + 1));
}

template <>
Dim<0> make_dim<0>(const int64_t* d) {
  return Dim<0>(*d);
}

void make_ddim(DDim& ddim, const int64_t* dims, int n) {
  switch (n) {
    case 0:
      ddim = make_dim<0>(dims);
      break;
    case 1:
      ddim = make_dim<1>(dims);
      break;
    case 2:
      ddim = make_dim<2>(dims);
      break;
    case 3:
      ddim = make_dim<3>(dims);
      break;
    case 4:
      ddim = make_dim<4>(dims);
      break;
    case 5:
      ddim = make_dim<5>(dims);
      break;
    case 6:
      ddim = make_dim<6>(dims);
      break;
    case 7:
      ddim = make_dim<7>(dims);
      break;
    case 8:
      ddim = make_dim<8>(dims);
      break;
    case 9:
      ddim = make_dim<9>(dims);
      break;
    default:
      PADDLE_THROW("Dynamic dimensions must have between [1, 9] dimensions.");
  }
}

/// @endcond

DDim make_ddim(std::initializer_list<int64_t> dims) {
  DDim result(make_dim(0));
  make_ddim(result, dims.begin(), dims.size());
  return result;
}

DDim make_ddim(const std::vector<int64_t>& dims) {
  DDim result(make_dim(0));
  make_ddim(result, &dims[0], dims.size());
  return result;
}

DDim make_ddim(const std::vector<int>& dims) {
  std::vector<int64_t> res(dims.size());
  std::transform(dims.begin(), dims.end(), res.begin(),
                 [](int d) { return static_cast<int64_t>(d); });
  return make_ddim(res);
}

/// @cond HIDDEN
// XXX For some reason, putting this in an anonymous namespace causes errors
class DynamicMutableIndexer : public boost::static_visitor<int64_t&> {
 public:
  explicit DynamicMutableIndexer(int idx) : idx_(idx) {}

  template <int D>
  int64_t& operator()(Dim<D>& dim) const {
    return dim[idx_];
  }

 private:
  int idx_;
};

class DynamicConstIndexer : public boost::static_visitor<int64_t> {
 public:
  explicit DynamicConstIndexer(int idx) : idx_(idx) {}

  template <int D>
  int64_t operator()(const Dim<D>& dim) const {
    return dim[idx_];
  }

 private:
  int idx_;
};

/// @endcond

int64_t& DDim::operator[](int idx) {
  return boost::apply_visitor(DynamicMutableIndexer(idx), var);
}

int64_t DDim::operator[](int idx) const {
  return boost::apply_visitor(DynamicConstIndexer(idx), var);
}

int DDim::size() const { return arity(*this); }

bool DDim::operator==(DDim d) const {
  if (var.which() != d.getVar().which()) {
    return false;
  } else {
    std::vector<int64_t> v1 = vectorize(*this);
    std::vector<int64_t> v2 = vectorize(d);

    for (unsigned int i = 0; i < v1.size(); i++) {
      if (v1[i] != v2[i]) {
        return false;
      }
    }

    return true;
  }
}

bool DDim::operator!=(DDim d) const { return !(*this == d); }

DDim DDim::operator+(DDim d) const {
  std::vector<int64_t> v1 = vectorize(*this);
  std::vector<int64_t> v2 = vectorize(d);

  std::vector<int64_t> v3;

  assert(v1.size() == v2.size());

  for (unsigned int i = 0; i < v1.size(); i++) {
    v3.push_back(v1[i] + v2[i]);
  }

  return make_ddim(v3);
}

DDim DDim::operator*(DDim d) const {
  std::vector<int64_t> v1 = vectorize(*this);
  std::vector<int64_t> v2 = vectorize(d);

  std::vector<int64_t> v3;

  assert(v1.size() == v2.size());

  for (unsigned int i = 0; i < v1.size(); i++) {
    v3.push_back(v1[i] * v2[i]);
  }

  return make_ddim(v3);
}

int64_t get(const DDim& ddim, int idx) { return ddim[idx]; }

void set(DDim& ddim, int idx, int value) { ddim[idx] = value; }

/// @cond HIDDEN
struct VectorizeVisitor : public boost::static_visitor<> {
  std::vector<int64_t>& vector;

  explicit VectorizeVisitor(std::vector<int64_t>& v) : vector(v) {}

  template <typename T>
  void operator()(const T& t) {
    vector.push_back(t.head);
    this->operator()(t.tail);
  }

  void operator()(const Dim<0>& t) {}
};
/// @endcond

std::vector<int64_t> vectorize(const DDim& ddim) {
  std::vector<int64_t> result;
  VectorizeVisitor visitor(result);
  boost::apply_visitor(visitor, ddim);
  return result;
}

// NOTE: framework::vectorize converts to type int64_t
//       which does not fit cudnn inputs.
std::vector<int> vectorize2int(const DDim& ddim) {
  std::vector<int64_t> temp = vectorize(ddim);
  std::vector<int> result(temp.begin(), temp.end());
  return result;
}

struct ProductVisitor : public boost::static_visitor<int64_t> {
  template <int D>
  int64_t operator()(const Dim<D>& dim) {
    return product(dim);
  }
};

int64_t product(const DDim& ddim) {
  ProductVisitor visitor;
  return boost::apply_visitor(visitor, ddim);
}

struct SliceVectorizeVisitor : public boost::static_visitor<> {
  std::vector<int64_t>& vector;
  int begin;
  int end;

  SliceVectorizeVisitor(std::vector<int64_t>& v, int b, int e)
      : vector(v), begin(b), end(e) {
    PADDLE_ENFORCE(begin < end,
                   "Begin index must be less than end index in ddim slice.");
    PADDLE_ENFORCE(begin >= 0,
                   "Begin index can't be less than zero in ddim slice.");
  }

  template <int S>
  void operator()(const Dim<S>& dim) {
    if (begin == 0) {
      vector.push_back(dim.head);
    } else {
      --begin;
    }
    --end;
    if (end > 0) {
      this->operator()(dim.tail);
    }
  }

  void operator()(const Dim<0>& dim) {
    PADDLE_ENFORCE(end == 0, "End index in ddim slice is out of bound.");
  }
};

DDim slice_ddim(const DDim& dim, int begin, int end) {
  std::vector<int64_t> vec;
  vec.reserve(end - begin);
  SliceVectorizeVisitor visitor(vec, begin, end);
  boost::apply_visitor(visitor, dim);
  return make_ddim(vec);
}

/// \cond HIDDEN

struct ArityVisitor : boost::static_visitor<int> {
  template <int D>
  int operator()(Dim<D>) const {
    return D;
  }
};

/// \endcond

int arity(const DDim& d) { return boost::apply_visitor(ArityVisitor(), d); }

/// \cond HIDDEN

struct DDimPrinter : boost::static_visitor<void> {
  std::ostream& os;
  explicit DDimPrinter(std::ostream& os_) : os(os_) {}

  template <typename T>
  void operator()(const T& t) {
    os << t;
  }
};

/// \endcond

std::ostream& operator<<(std::ostream& os, const DDim& ddim) {
  DDimPrinter printer(os);
  boost::apply_visitor(printer, ddim);
  return os;
}

DDim::DDim(std::initializer_list<int64_t> init_list) {
  *this = make_ddim(init_list);
}

DDim flatten_to_2d(const DDim& src, int num_col_dims) {
  int rank = src.size();
  return make_ddim({product(slice_ddim(src, 0, num_col_dims)),
                    product(slice_ddim(src, num_col_dims, rank))});
}

DDim flatten_to_1d(const DDim& src) { return make_ddim({product(src)}); }

DDim stride(const DDim& ddim) {
  std::vector<int64_t> strides(ddim.size());
  strides[ddim.size() - 1] = 1;
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i + 1];
  }
  return framework::make_ddim(strides);
}

DDim stride_numel(const framework::DDim& ddim) {
  std::vector<int64_t> strides(ddim.size());
  strides[ddim.size() - 1] = ddim[ddim.size() - 1];
  for (int i = ddim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * ddim[i];
  }
  return framework::make_ddim(strides);
}

}  // namespace framework
}  // namespace paddle
