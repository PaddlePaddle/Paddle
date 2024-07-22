//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/phi/common/place.h"

#include "paddle/common/ddim.h"

namespace paddle {
namespace framework {

TEST(EigenDim, From) {
  EigenDim<3>::Type ed = EigenDim<3>::From(common::make_ddim({1, 2, 3}));
  ASSERT_EQ(1, ed[0]);
  ASSERT_EQ(2, ed[1]);
  ASSERT_EQ(3, ed[2]);
}

TEST(Eigen, DenseTensor) {
  phi::DenseTensor t;
  float* p =
      t.mutable_data<float>(common::make_ddim({1, 2, 3}), phi::CPUPlace());
  for (int i = 0; i < 1 * 2 * 3; i++) {
    p[i] = static_cast<float>(i);
  }

  EigenTensor<float, 3>::Type et = EigenTensor<float, 3>::From(t);

  ASSERT_EQ(1, et.dimension(0));
  ASSERT_EQ(2, et.dimension(1));
  ASSERT_EQ(3, et.dimension(2));

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 3; k++) {
        ASSERT_NEAR((i * 2 + j) * 3 + k, et(i, j, k), 1e-6f);
      }
    }
  }
}

TEST(Eigen, ScalarFrom) {
  phi::DenseTensor t;
  int* p = t.mutable_data<int>(common::make_ddim({1}), phi::CPUPlace());
  *p = static_cast<int>(100);

  EigenScalar<int>::Type es = EigenScalar<int>::From(t);

  ASSERT_EQ(0, es.dimension(0));
  ASSERT_EQ(100, es(0));
}

TEST(Eigen, VectorFrom) {
  phi::DenseTensor t;
  float* p = t.mutable_data<float>(common::make_ddim({6}), phi::CPUPlace());
  for (int i = 0; i < 6; i++) {
    p[i] = static_cast<float>(i);
  }

  EigenVector<float>::Type ev = EigenVector<float>::From(t);

  ASSERT_EQ(6, ev.dimension(0));

  for (int i = 0; i < 6; i++) {
    ASSERT_NEAR(i, ev(i), 1e-6f);
  }
}

TEST(Eigen, VectorFlatten) {
  phi::DenseTensor t;
  float* p =
      t.mutable_data<float>(common::make_ddim({1, 2, 3}), phi::CPUPlace());
  for (int i = 0; i < 1 * 2 * 3; i++) {
    p[i] = static_cast<float>(i);
  }

  EigenVector<float>::Type ev = EigenVector<float>::Flatten(t);

  ASSERT_EQ(1 * 2 * 3, ev.dimension(0));

  for (int i = 0; i < 1 * 2 * 3; i++) {
    ASSERT_NEAR(i, ev(i), 1e-6f);
  }
}

TEST(Eigen, Matrix) {
  phi::DenseTensor t;
  float* p = t.mutable_data<float>(common::make_ddim({2, 3}), phi::CPUPlace());
  for (int i = 0; i < 2 * 3; i++) {
    p[i] = static_cast<float>(i);
  }

  EigenMatrix<float>::Type em = EigenMatrix<float>::From(t);

  ASSERT_EQ(2, em.dimension(0));
  ASSERT_EQ(3, em.dimension(1));

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      ASSERT_NEAR(i * 3 + j, em(i, j), 1e-6f);
    }
  }
}

TEST(Eigen, MatrixReshape) {
  phi::DenseTensor t;
  float* p = t.mutable_data<float>({2, 3, 6, 4}, phi::CPUPlace());
  for (int i = 0; i < 2 * 3 * 6 * 4; ++i) {
    p[i] = static_cast<float>(i);
  }

  EigenMatrix<float>::Type em = EigenMatrix<float>::Reshape(t, 2);

  ASSERT_EQ(2 * 3, em.dimension(0));
  ASSERT_EQ(6 * 4, em.dimension(1));

  for (int i = 0; i < 2 * 3; i++) {
    for (int j = 0; j < 6 * 4; j++) {
      ASSERT_NEAR(i * 6 * 4 + j, em(i, j), 1e-6f);
    }
  }
}

}  // namespace framework
}  // namespace paddle
