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

#pragma once

#include <time.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "grpc++/grpc++.h"
#include "grpc++/support/byte_buffer.h"
#include "grpc++/support/slice.h"
#include "grpc/support/log.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/detail/request_handler.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace detail {

class GrpcRequestSendHandler final : public RequestHandler {
 public:
  explicit GrpcRequestSendHandler(bool sync_mode) : RequestHandler(sync_mode) {}
  virtual ~GrpcRequestSendHandler() {}
  bool Handle(void* input, void* output) override;
};

class GrpcRequestGetHandler final : public RequestHandler {
 public:
  explicit GrpcRequestGetHandler(bool sync_mode) : RequestHandler(sync_mode) {}
  virtual ~GrpcRequestGetHandler() {}
  bool Handle(void* input, void* output) override;
};

class GrpcRequestPrefetchHandler final : public RequestHandler {
 public:
  explicit GrpcRequestPrefetchHandler(bool sync_mode)
      : RequestHandler(sync_mode) {}
  virtual ~GrpcRequestPrefetchHandler() {}
  bool Handle(void* input, void* output) override;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
