/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <sstream>
#include "paddle/topology/serialization/Archiver.h"

TEST(JsonArchiver, serialize) {
  auto& archive = paddle::topology::serialization::IArchiver::JsonArchiver();
  paddle::topology::AttributeMap attrs;
  attrs["testInt"] = 10;
  attrs["testDouble"] = (double)2.34;
  attrs["testString"] = std::string("abcdefg");
  attrs["testStdVectorInt"] = std::vector<int>{1, 2, 3, 4, 5};
  attrs["testStdVectorSizeT"] = std::vector<size_t>{3, 4, 5, 6, 1};
  std::ostringstream sout;

  auto err = archive.serialize(attrs, sout);
  ASSERT_TRUE(err.isOK());

  paddle::topology::meta::AttributeMetaMap metaMap;
  metaMap.addAttribute<int>("testInt", "");
  metaMap.addAttribute<double>("testDouble", "");
  metaMap.addAttribute<std::vector<int>>("testStdVectorInt", "");
  metaMap.addAttribute<std::vector<size_t>>("testStdVectorSizeT", "");
  metaMap.addAttribute<std::string>("testString", "");

  paddle::topology::AttributeMap attrs2;
  std::istringstream sin(sout.str());
  err = archive.deserialize(sin, metaMap, &attrs2);
  ASSERT_TRUE(err.isOK());
  std::ostringstream sout2;
  err = archive.serialize(attrs, sout2);
  ASSERT_TRUE(err.isOK());
  ASSERT_EQ(sout.str(), sout2.str());
}
