/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/framework/details/cow_ptr.h"
#include "gtest/gtest.h"

namespace paddle {
namespace framework {
namespace details {

TEST(COWPtr, all) {
  COWPtr<int> ptr(new int{0});
  ASSERT_EQ(ptr.Data(), 0);
  COWPtr<int> ptr2 = ptr;
  ASSERT_EQ(ptr2.Data(), 0);
  ASSERT_EQ(&ptr2.Data(), &ptr.Data());
  *ptr2.MutableData() = 10;
  ASSERT_EQ(ptr.Data(), 0);
  ASSERT_EQ(ptr2.Data(), 10);
}

TEST(COWPtr, change_old) {
  COWPtr<int> ptr(new int{0});
  COWPtr<int> ptr2 = ptr;
  *ptr.MutableData() = 10;
  ASSERT_EQ(ptr2.Data(), 0);
  ASSERT_EQ(ptr.Data(), 10);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
