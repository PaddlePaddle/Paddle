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

#include "paddle/contrib/tape/variable.h"

namespace paddle {
namespace tape {

void Variable::InitializeVariable() {
  LOG(INFO) << "Initialzing " << desc_.Name() << " as " << desc_.GetType();
  framework::proto::VarType::Type var_type = desc_.GetType();
  if (var_type == framework::proto::VarType::LOD_TENSOR) {
    var_.GetMutable<framework::LoDTensor>();
  } else if (var_type == framework::proto::VarType::SELECTED_ROWS) {
    var_.GetMutable<framework::SelectedRows>();
  } else {
    PADDLE_THROW("Variable type %d is not in [LOD_TENSOR, SELECTED_ROWS]",
                 var_type);
  }
}
}
}
