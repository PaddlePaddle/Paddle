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
#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/prune.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
void save_model(const std::unique_ptr<ProgramDesc>& main_program, Scope* scope,
                const std::vector<std::string>& param_names,
                const std::string& model_name, bool save_combine);
}
}
