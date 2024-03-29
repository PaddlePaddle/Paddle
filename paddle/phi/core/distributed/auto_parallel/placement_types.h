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

// This API design is inspired by:
// https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/placement_types.py
// Git commit hash: 52e2b87d00ed527dc7f990d1a7a4c5498f99c513

#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/phi/common/reduce_type.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {
namespace distributed {

class Placement {
 public:
  virtual ~Placement() = default;

  virtual bool is_shard(std::optional<int> dim = std::nullopt) const {
    return false;
  }

  virtual bool is_replicated() const { return false; }

  virtual bool is_partial() const { return false; }

  virtual size_t hash() const { return 0; }

  virtual std::string to_string() const { return ""; }

  virtual bool operator==(const Placement& other) const {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Equal function is not implemented yet in Placement."));
  }

  virtual bool operator!=(const Placement& other) const {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Not Equal function is not implemented yet in Placement."));
  }

  friend std::ostream& operator<<(std::ostream& os, const Placement& p) {
    os << p.to_string();
    return os;
  }
};

class Shard : public Placement {
 public:
  explicit Shard(int dim) : dim_(dim) {}

  bool is_shard(std::optional<int> dim = std::nullopt) const override {
    if (dim && *dim == this->dim_) {
      return true;
    } else {
      return !dim.has_value();
    }
  }

  bool operator==(const Placement& other) const override {
    const Shard* other_shard = dynamic_cast<const Shard*>(&other);
    return other_shard && this->dim_ == other_shard->dim_;
  }

  bool operator!=(const Placement& other) const override {
    return !(*this == other);
  }

  std::size_t hash() const override {
    return std::hash<std::string>{}(to_string());
  }

  int get_dim() const { return dim_; }

  friend std::ostream& operator<<(std::ostream& os, const Shard& p) {
    os << p.to_string();
    return os;
  }

  std::string to_string() const override {
    return "Shard(dim=" + std::to_string(dim_) + ")";
  }

 private:
  int dim_;
};

class Replicate : public Placement {
 public:
  bool is_replicated() const override { return true; }

  bool operator==(const Placement& other) const override {
    return dynamic_cast<const Replicate*>(&other) != nullptr;
  }

  bool operator!=(const Placement& other) const override {
    return !(*this == other);
  }

  std::size_t hash() const override {
    return std::hash<std::string>{}(to_string());
  }

  friend std::ostream& operator<<(std::ostream& os, const Replicate& p) {
    os << p.to_string();
    return os;
  }

  std::string to_string() const override { return "Replicate()"; }
};

class Partial : public Placement {
 public:
  explicit Partial(ReduceType reduce_type) : reduce_type_(reduce_type) {}
  bool is_partial() const override { return true; }
  ReduceType get_reduce_type() const { return reduce_type_; }

  bool operator==(const Placement& other) const override {
    const Partial* other_partial = dynamic_cast<const Partial*>(&other);
    return other_partial && this->reduce_type_ == other_partial->reduce_type_;
  }

  bool operator!=(const Placement& other) const override {
    return !(*this == other);
  }

  std::size_t hash() const override {
    return std::hash<std::string>{}(to_string());
  }

  friend std::ostream& operator<<(std::ostream& os, const Partial& p) {
    os << p.to_string();
    return os;
  }

  std::string to_string() const override {
    return "Partial(reduce_type=" +
           std::string(ReduceTypeStrings[static_cast<int>(reduce_type_)]) + ")";
  }

 private:
  ReduceType reduce_type_;
};

using Placements = std::vector<std::shared_ptr<Placement>>;
class DistTensorMeta : public std::enable_shared_from_this<DistTensorMeta> {
 public:
  DistTensorMeta(const ProcessMesh& process_mesh,
                 const Placements& placements,
                 const DenseTensorMeta& tensor_meta)
      : process_mesh_(std::make_shared<const ProcessMesh>(process_mesh)),
        placements_(placements),
        tensor_meta_(std::make_shared<const DenseTensorMeta>(tensor_meta)) {}

  DistTensorMeta() = default;

  const DDim& dims() const { return tensor_meta_->dims; }

  const ProcessMesh& process_mesh() const { return *process_mesh_; }

  const Placements& placements() const { return placements_; }

  int64_t num_shard() const;

  std::vector<int64_t> dim_mapping() const;

  bool is_replicated() const;

 private:
  std::shared_ptr<const ProcessMesh> process_mesh_;
  Placements placements_;
  std::shared_ptr<const DenseTensorMeta> tensor_meta_;
};

}  // namespace distributed
}  // namespace phi

namespace std {
template <>
struct hash<phi::distributed::Placement> {
  std::size_t operator()(const phi::distributed::Placement& p) const {
    return p.hash();
  }
};

}  // namespace std
