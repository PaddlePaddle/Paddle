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

#pragma once
#include <iostream>
#include <sstream>
#include <unordered_map>
#include "paddle/topology/AttributeMap.h"
#include "paddle/topology/meta/AttributeMeta.h"

namespace paddle {
namespace topology {
namespace serialization {

/**
 * @brief The IArchiver class
 */
class IArchiver {
public:
  virtual ~IArchiver();

  virtual Error __must_check serialize(const AttributeMap& attrs,
                                       std::ostream& sout) = 0;

  Error __must_check serialize(const AttributeMap& attrs, std::string& out) {
    std::ostringstream sout;
    auto err = this->serialize(attrs, sout);
    out = sout.str();
    return err;
  }

  virtual Error __must_check deserialize(std::istream& in,
                                         const meta::AttributeMetaMap& meta,
                                         AttributeMap* attrs) = 0;

  Error __must_check deserialize(const std::string& str,
                                 const meta::AttributeMetaMap& meta,
                                 AttributeMap* attrs) {
    std::istringstream sin(str);
    return deserialize(sin, meta, attrs);
  }

  static IArchiver& JsonArchiver();
};

}  // namespace serialization
}  // namespace topology
}  // namespace paddle
