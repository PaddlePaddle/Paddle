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
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "gtest/gtest.h"

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/group.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace ir {

void BuildNoCircleGraph(Graph* g) {
  OpDesc op1;
  op1.SetType("conv2d");
  OpDesc op2;
  op2.SetType("relu");
  OpDesc op3;
  op3.SetType("relu");
  OpDesc op4;
  op4.SetType("sigmoid");
  OpDesc op5;
  op5.SetType("elementwise_add");
  VarDesc var1("var1");
  VarDesc var2("var2");
  VarDesc var3("var3");
  VarDesc var4("var4");

  ir::Node* o1 = g->CreateOpNode(&op1);
  ir::Node* o2 = g->CreateOpNode(&op2);
  ir::Node* o3 = g->CreateOpNode(&op3);
  ir::Node* o4 = g->CreateOpNode(&op4);
  ir::Node* o5 = g->CreateOpNode(&op5);
  ir::Node* v1 = g->CreateVarNode(&var1);
  ir::Node* v2 = g->CreateVarNode(&var2);
  ir::Node* v3 = g->CreateVarNode(&var3);
  ir::Node* v4 = g->CreateVarNode(&var4);

  // o1->v1->o2
  o1->outputs.push_back(v1);
  o2->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o2);
  // o2->v2->o3
  // o2->v2->o4
  o2->outputs.push_back(v2);
  o3->inputs.push_back(v2);
  o4->inputs.push_back(v2);
  v2->outputs.push_back(o3);
  v2->outputs.push_back(o4);
  v2->inputs.push_back(o2);
  // o4->v3->o5
  o4->outputs.push_back(v3);
  o5->inputs.push_back(v3);
  v3->inputs.push_back(o4);
  v3->outputs.push_back(o5);
  // o3-v4->o5
  o3->outputs.push_back(v4);
  o5->inputs.push_back(v4);
  v4->inputs.push_back(o3);
  v4->outputs.push_back(o5);
}

TEST(GroupTest, BasicFuse) {
  ProgramDesc prog;
  ElementwiseGroupDetector detector;
  Graph test_graph(prog);
  BuildNoCircleGraph(&test_graph);
  detector.MarkFusedGroup(test_graph);
  std::vector<Group> result = detector.GetFusedGroups();
  for (size_t i = 0; i < result.size(); i++) {
    result[i].PrintGroup();
    std::cout << "--------------" << std::endl;
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
