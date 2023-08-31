/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

// General bottom line rules

SpmdInfo ReplicatedSpmdInferForward(
    const std::vector<const DistMetaTensor*>& inputs,
    const std::vector<const DistMetaTensor*>& outputs,
    const std::vector<phi::Attribute>& attrs);

SpmdInfo ReplicatedSpmdInferBackward(
    const std::vector<const DistMetaTensor*>& inputs,
    const std::vector<const DistMetaTensor*>& outputs,
    const std::vector<phi::Attribute>& attrs);

namespace detail {

// Adapt variadic argument
template <typename Functor>
struct ArgsIterator {
  template <typename... Args>
  inline Functor& apply() {
    return self();
  }

  template <typename T, typename... Args>
  inline Functor& apply(T&& arg, Args&&... args) {
    self()(std::forward<T>(arg));
    if (self().short_circuit()) {
      return self();
    } else {
      return apply(std::forward<Args>(args)...);
    }
  }

  constexpr bool short_circuit() const { return false; }

 private:
  inline Functor& self() { return *static_cast<Functor*>(this); }
};

struct ReplicatedSpmdArgumentParser
    : public ArgsIterator<ReplicatedSpmdArgumentParser> {
  std::vector<const DistMetaTensor*> inputs;
  std::vector<const DistMetaTensor*> outputs;
  std::vector<phi::Attribute> attrs;

  // deal with inputs
  void operator()(const DistMetaTensor& x) { inputs.emplace_back(&x); }

  void operator()(const std::vector<const DistMetaTensor*>& x) {
    for (auto t : x) {
      inputs.emplace_back(t);
    }
  }

  template <typename AttrType>
  void operator()(AttrType x) {
    attrs.emplace_back(x);
  }

  // deal with outputs
  void operator()(DistMetaTensor* out) { outputs.emplace_back(out); }

  void operator()(std::vector<DistMetaTensor*> out) {
    for (auto t : out) {
      outputs.emplace_back(t);
    }
  }

  SpmdInfo InferForward() {
    return ReplicatedSpmdInferForward(inputs, outputs, attrs);
  }

  SpmdInfo InferBackward() {
    return ReplicatedSpmdInferBackward(inputs, outputs, attrs);
  }
};

}  // namespace detail

// For generated phi api
template <typename... Args>
SpmdInfo GeneralSpmdInferForward(const Args&... args) {
  return detail::ReplicatedSpmdArgumentParser().apply(args...).InferForward();
}

template <typename... Args>
SpmdInfo GeneralSpmdInferBackward(const Args&... args) {
  return detail::ReplicatedSpmdArgumentParser().apply(args...).InferBackward();
}

}  // namespace distributed
}  // namespace phi
