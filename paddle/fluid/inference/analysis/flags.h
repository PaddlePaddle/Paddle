// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <gflags/gflags.h>

// TODO(Superjomn) add a definition flag like PADDLE_WITH_TENSORRT and hide this
// flag if not available.
DECLARE_bool(IA_enable_tensorrt_subgraph_engine);
DECLARE_string(IA_graphviz_log_root);
DECLARE_string(IA_output_storage_path);
DECLARE_bool(IA_enable_ir);
