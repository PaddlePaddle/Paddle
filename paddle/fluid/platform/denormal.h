// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace platform {

// Used to restore the initial value at the end of the scope.
class ScopedRestoreFlushDenormalState {
 public:
  ScopedRestoreFlushDenormalState();
  ~ScopedRestoreFlushDenormalState();

 private:
  bool flush_zero_mode_;
  bool denormals_zero_mode_;
  DISABLE_COPY_AND_ASSIGN(ScopedRestoreFlushDenormalState);
};

class ScopedFlushDenormal {
 public:
  ScopedFlushDenormal();

 private:
  ScopedRestoreFlushDenormalState restore_;
  DISABLE_COPY_AND_ASSIGN(ScopedFlushDenormal);
};
}  // namespace platform
}  // namespace paddle
