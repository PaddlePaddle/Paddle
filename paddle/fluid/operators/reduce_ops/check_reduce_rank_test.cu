// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"
#include "paddle/pten/kernels/gpu/reduce.h"

namespace paddle {
namespace operators {
namespace details {

TEST(test_reduce_rank_check, all) {
  using EnforceNotMet = paddle::platform::EnforceNotMet;
  constexpr int kMaxRank = framework::DDim::kMaxRank;

  for (int rank = 0; rank < kMaxRank; rank++) {
    for (int reduce_rank = 0; reduce_rank <= rank; reduce_rank++) {
      bool is_valid = false;
      if (rank % 2 == 0) {
        is_valid = (reduce_rank == rank / 2);
      } else {
        if (reduce_rank == (rank - 1) / 2) {
          is_valid = true;
        } else if (reduce_rank == (rank + 1) / 2) {
          is_valid = true;
        } else {
          is_valid = false;
        }
      }

      if (is_valid) {
        pten::kernels::details::CheckReduceRank(reduce_rank, rank);
      } else {
        ASSERT_THROW(pten::kernels::details::CheckReduceRank(reduce_rank, rank),
                     paddle::platform::EnforceNotMet);
      }
    }
  }
}

}  // namespace details
}  // namespace operators
}  // namespace paddle
