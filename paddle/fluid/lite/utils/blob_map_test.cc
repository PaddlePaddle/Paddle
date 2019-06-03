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

#include "paddle/fluid/lite/utils/blob_map.h"
#include <gtest/gtest.h>

namespace paddle {
namespace lite {

TEST(BlobMap, test) {
  BlobMap map0("./1.txt");
  map0.Insert("a", "xxx");
  ASSERT_TRUE(map0.Persist());

  BlobMap map1("./1.txt");
  ASSERT_TRUE(map1.Load());
}

}  // namespace lite
}  // namespace paddle
