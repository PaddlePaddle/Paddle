//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/tensor_py.h"

#include <iostream>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/tensor.h"

TEST(TensorPy, CastToPyBufferImpl) {
  typedef int ElemType;

  paddle::framework::Tensor t;
  auto d = paddle::framework::make_ddim({1, 2, 3});
  int* p = t.mutable_data<ElemType>(d, paddle::platform::CPUPlace());
  for (int i = 0; i < paddle::framework::product(d); ++i) {
    p[i] = i;
  }

  pybind11::buffer_info bi = paddle::pybind::CastToPyBuffer(t);
  EXPECT_EQ(bi.itemsize, static_cast<size_t>(sizeof(ElemType)));
  EXPECT_EQ(bi.size, static_cast<size_t>(paddle::framework::product(d)));
  EXPECT_EQ(bi.ndim, static_cast<size_t>(3));  // 3-dimensional as d.
  EXPECT_EQ(bi.shape.size(), 3U);              // as Dim d.
  EXPECT_EQ(bi.shape[0], static_cast<size_t>(1));
  EXPECT_EQ(bi.shape[1], static_cast<size_t>(2));
  EXPECT_EQ(bi.shape[2], static_cast<size_t>(3));
  EXPECT_EQ(bi.strides.size(), 3U);  // 3-dimensional as d.
  EXPECT_EQ(bi.strides[2], static_cast<size_t>(sizeof(ElemType)));
  EXPECT_EQ(bi.strides[1], static_cast<size_t>(sizeof(ElemType) * 3));
  EXPECT_EQ(bi.strides[0], static_cast<size_t>(sizeof(ElemType) * 2 * 3));
}


//TEST(TensorPy, PySliceTensor) {
//  typedef int64_t ElemType;
//
//  paddle::framework::Tensor t;
//  auto d = paddle::framework::make_ddim({3, 3, 3});
//  int64_t* p = t.mutable_data<ElemType>(d, paddle::platform::CPUPlace());
//  for (int i = 0; i < paddle::framework::product(d); ++i) {
//    p[i] = i;
//  }
//
//auto a =py::list();
//a.append(0);
//a.append(0);
//
//   paddle::framework::Tensor s2 = *paddle::pybind::PySliceTensor(t,
////																																 *new py::list({py::cast(0)}));
//a);
////  paddle::framework::Tensor s2 =
////     *paddle::pybind::PySliceTensor(t, *new py::slice(0, -1, 2));
////paddle::framework::Tensor s2 =
////   *paddle::pybind::_sliceTensorByDim(t, 0, 0, 1, 1, 1);
//
//  pybind11::buffer_info bi = paddle::pybind::CastToPyBuffer(s2);
//  EXPECT_EQ(bi.itemsize, static_cast<size_t>(sizeof(ElemType)));
//  EXPECT_EQ(bi.size, static_cast<size_t>(paddle::framework::product(
//                         paddle::framework::make_ddim({2, 3, 3}))));
//  EXPECT_EQ(bi.ndim, static_cast<size_t>(3));  // 3-dimensional as d.
//  EXPECT_EQ(bi.shape.size(), 3U);              // as Dim d.
//  EXPECT_EQ(bi.shape[0], static_cast<size_t>(2));
//  EXPECT_EQ(bi.shape[1], static_cast<size_t>(3));
//  EXPECT_EQ(bi.shape[2], static_cast<size_t>(3));
//  EXPECT_EQ(bi.strides.size(), 3U);  // 3-dimensional as d.
//  EXPECT_EQ(bi.strides[2], static_cast<size_t>(sizeof(ElemType)));
//  EXPECT_EQ(bi.strides[1], static_cast<size_t>(sizeof(ElemType) * 3));
//  EXPECT_EQ(bi.strides[0], static_cast<size_t>(sizeof(ElemType) * 3 * 3));
//}