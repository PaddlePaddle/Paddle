/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

/*
1. before fuse:
  div
   |
  cast-----------|-------------|
   |             |             |
yolo_box      yolo_box      yolo_box
   |             |             |
transpose-|   transpose-|   transpose-|
   |------|-----|-------|------|      |
          |   concat    |             |
          |-----|-------|-------------|
                |     concat
                |-------|
                       nms3

2. after fuse:
yolo_box_head      yolo_box_head      yolo_box_head
      |------------------|------------------|
                    yolo_box_post
*/
class YoloBoxFusePass : public FusePassBase {
 public:
  YoloBoxFusePass();
  virtual ~YoloBoxFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  std::string name_scope_{"yolo_box_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
