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
#pragma once

#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/tcp/device.h>

#include <memory>

#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/macros.h"

namespace phi {
class DenseTensor;
namespace distributed {

class GlooCommContext final : public CommContext {
 public:
  GlooCommContext(int rank,
                  int size,
                  std::shared_ptr<gloo::rendezvous::Store> store,
                  std::shared_ptr<gloo::transport::Device> device);

  void Broadcast(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 int root);
  void AllReduce(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor,
                 int reduce_type);
  void Reduce(phi::DenseTensor* out_tensor,
              const phi::DenseTensor& in_tensor,
              int reduce_type,
              int root);

  void AllGather(phi::DenseTensor* out_tensor,
                 const phi::DenseTensor& in_tensor);

 private:
  DISABLE_COPY_AND_ASSIGN(GlooCommContext);

  std::shared_ptr<gloo::rendezvous::Context> gloo_context_;
};

}  // namespace distributed
}  // namespace phi
