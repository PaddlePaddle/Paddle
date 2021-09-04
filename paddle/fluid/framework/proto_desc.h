/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

namespace paddle {
namespace framework {

// The Index of first Block in Program. also called root block.
constexpr int kRootBlockIndex = 0;
// The Parent Index of root Block, this block does not exist.
constexpr int kNoneBlockIndex = -1;

// The Parent Index of root ProcessMesh, this ProcessMesh does not exist.
constexpr int kNoneProcessMeshIndex = -1;

// If a attribute name has a certain suffix, it means that the
// atrribute is a distributed-related attribute for auto parallel.
// e.g., "mesh_id@PARALLEL".
constexpr char kAutoParallelSuffix[] = "@PARALLEL";

}  // namespace framework
}  // namespace paddle
