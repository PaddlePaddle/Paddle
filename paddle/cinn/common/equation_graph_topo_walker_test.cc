// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

// TODO(yifan): Add unittest here
#include "paddle/cinn/common/equation_graph_topo_walker.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace adt {
namespace common {

using VT = int;
using FT = std::string;

TEST(EquationGraphTopoWalker, simple1) {
  auto F4V = [](VT variable, const std::function<void(FT)>& visitor) {
    if (variable == 1) {
      visitor("b");
    } else if (variable == 3) {
      visitor("d");
    }
  };
  auto InV4F = [](FT function, const std::function<void(VT)>& visitor) {
    if (function == "b") {
      visitor(1);
    } else if (function == "d") {
      visitor(3);
    }
  };
  auto OutV4F = [](FT function, const std::function<void(VT)>& visitor) {
    if (function == "b" || function == "d") {
      visitor(567);
    } else if (function == "e") {
      visitor(890);
      visitor(101112);
    }
  };
  cinn::EquationGraphTopoWalker<VT, FT> walker(F4V, InV4F, OutV4F);
  std::vector<FT> outputs;
  std::function<void(FT)> functionVisitor = [&](FT function) {
    LOG(ERROR) << function;
    outputs.push_back(function);
  };
  walker(1, functionVisitor);
  std::vector<FT> expected{"b"};
  EXPECT_TRUE((outputs == expected));
}

TEST(EquationGraphTopoWalker, simple2) {
  auto F4V = [](VT variable, const std::function<void(FT)>& visitor) {
    if (variable == 1) {
      visitor("bcd");
    } else if (variable == 3) {
      visitor("d");
    } else if (variable == 567) {
      visitor("jkl");
    }
  };
  auto InV4F = [](FT function, const std::function<void(VT)>& visitor) {
    if (function == "bcd") {
      visitor(1);
    } else if (function == "d") {
      visitor(3);
    }
  };
  auto OutV4F = [](FT function, const std::function<void(VT)>& visitor) {
    if (function == "bcd" || function == "d") {
      visitor(567);
    } else if (function == "e") {
      visitor(890);
      visitor(101112);
    }
  };
  cinn::EquationGraphTopoWalker<VT, FT> walker(F4V, InV4F, OutV4F);
  std::vector<VT> outputs;
  std::function<void(VT)> variableVisitor = [&](VT variable) {
    outputs.push_back(variable);
  };
  walker(1, variableVisitor);
  std::vector<VT> expected{1, 567};
  EXPECT_TRUE((outputs == expected));
}

}  // namespace common
}  // namespace adt
