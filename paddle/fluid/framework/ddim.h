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

#pragma once

#include <initializer_list>
#include <stdexcept>
#include <vector>
#include "paddle/fluid/framework/dim.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {

/**
 * \brief A dynamically sized dimension.
 *
 * The number of dimensions must be between [1, 9].
 */
struct DDim {
  typedef boost::variant<Dim<0>, Dim<1>, Dim<2>, Dim<3>, Dim<4>, Dim<5>, Dim<6>,
                         Dim<7>, Dim<8>, Dim<9>>
      DDimVar;
  DDimVar var;

  DDim() : var(Dim<1>()) {}

  template <int D>
  explicit DDim(const Dim<D>& in) : var(in) {}

  /*implicit*/ DDim(std::initializer_list<int64_t> init_list);

  template <int D>
  DDim& operator=(const Dim<D>& in) {
    var = in;
    return *this;
  }

  int64_t& operator[](int idx);
  int64_t operator[](int idx) const;

  template <typename Visitor>
  typename Visitor::result_type apply_visitor(Visitor& visitor) {
    return var.apply_visitor(visitor);
  }

  template <typename Visitor>
  typename Visitor::result_type apply_visitor(Visitor& visitor) const {
    return var.apply_visitor(visitor);
  }

  DDimVar getVar() { return var; }

  bool operator==(DDim d) const;

  bool operator!=(DDim d) const;

  DDim operator+(DDim d) const;

  DDim operator*(DDim d) const;

  int size() const;
};

/**
 * \brief Make a DDim from std::vector<int64_t>
 *
 * \param dims An vector of ints. Must be sized between [1, 9]
 */
DDim make_ddim(const std::vector<int64_t>& dims);

DDim make_ddim(const std::vector<int>& dims);

/**
 * \brief Make a DDim from an initializer list
 *
 * \param dims An initializer list of ints. Must be sized between [1, 9]
 *
 */
DDim make_ddim(std::initializer_list<int64_t> dims);

int64_t get(const DDim& dim, int idx);
void set(DDim& dim, int idx, int val);

std::vector<int64_t> vectorize(const DDim& ddim);
std::vector<int> vectorize2int(const DDim& ddim);

int64_t product(const DDim& ddim);

/**
 * \brief Slice a ddim
 *
 * Slice dim with [begin, end).
 * e.g.  DDim d = make_ddim({1,2,3,4,5});
 *       slice_ddim(d, 1, 3); ====> {2,3}
 */
DDim slice_ddim(const DDim& dim, int begin, int end);

/**
 * \brief What is the length of this dimension?
 *
 * \param Dynamic dimension to inspect
 */

int arity(const DDim& ddim);

std::ostream& operator<<(std::ostream&, const DDim&);

// Reshape a tensor to a matrix. The matrix's first dimension(column length)
// will be the product of tensor's first `num_col_dims` dimensions.
DDim flatten_to_2d(const DDim& src, int num_col_dims);

DDim flatten_to_1d(const DDim& src);

DDim stride(const DDim& ddim);

DDim stride_numel(const DDim& ddim);
}  // namespace framework
}  // namespace paddle

namespace boost {

template <typename T>
T get(const paddle::framework::DDim& in) {
  return boost::get<T>(in.var);
}

}  // namespace boost
