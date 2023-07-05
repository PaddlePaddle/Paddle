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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dim_trans.h"
#include <assert.h>
#include <cstdio>
#include <numeric>
#include <set>
#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

static std::vector<DimTrans*> all_dim_trans;

DimTrans::DimTrans(Type type) : type_(type) {}

DimTrans::~DimTrans() {}

DimTrans::Type DimTrans::type() const { return type_; }

void DimTrans::set_type(Type type) { type_ = type; }

void DimTrans::print_info() {}

InputDim::InputDim() : DimTrans(DimTrans::Type::INPUTDIM) {
  input_dim_ = -1;
  all_dim_trans.emplace_back(this);
}

InputDim::InputDim(int64_t dim) : DimTrans(DimTrans::Type::INPUTDIM) {
  input_dim_ = dim;
  all_dim_trans.emplace_back(this);
}

InputDim::~InputDim() {}

int64_t InputDim::input_dim() const { return input_dim_; }

void InputDim::set_input_dim(int64_t dim) { input_dim_ = dim; }

void InputDim::print_info() { printf("InputDim(%ld)", input_dim_); }

Singleton::Singleton() : DimTrans(DimTrans::Type::SINGLETON) {
  all_dim_trans.emplace_back(this);
}

void Singleton::print_info() { printf("Singleton()"); }

Flatten::Flatten() : DimTrans(DimTrans::Type::FLATTEN) {
  all_dim_trans.emplace_back(this);
}

Flatten::Flatten(const std::vector<DimTrans*>& dims)
    : DimTrans(DimTrans::Type::FLATTEN) {
  input_dims_ = dims;
  all_dim_trans.emplace_back(this);
}

Flatten::~Flatten() {
  input_dims_.assign(input_dims_.size(), nullptr);
  std::vector<DimTrans*>().swap(input_dims_);
}

const std::vector<DimTrans*>& Flatten::inputs() const { return input_dims_; }

void Flatten::set_inputs(const std::vector<DimTrans*>& dims) {
  input_dims_.assign(dims.begin(), dims.end());
}

void Flatten::print_info() {
  printf("Flatten(");
  for (int64_t i = 0, n = input_dims_.size(); i < n; ++i) {
    input_dims_[i]->print_info();
    if (i < n - 1) {
      printf(",");
    }
  }
  printf(")");
}

Split::Split() : DimTrans(DimTrans::Type::SPLIT) {
  input_dim_trans_ = nullptr;
  all_dim_trans.emplace_back(this);
}

Split::Split(DimTrans* dim, const std::vector<int64_t>& shape, int64_t id)
    : DimTrans(DimTrans::Type::SPLIT) {
  input_dim_trans_ = dim;
  split_id_ = id;
  splitted_shape_.assign(shape.begin(), shape.end());
  all_dim_trans.emplace_back(this);
}

Split::~Split() {
  input_dim_trans_ = nullptr;
  std::vector<int64_t>().swap(splitted_shape_);
}

DimTrans* Split::input() const { return input_dim_trans_; }

void Split::set_input(DimTrans* dim) { input_dim_trans_ = dim; }

int64_t Split::split_id() const { return split_id_; }

int64_t Split::local_split_shape() { return splitted_shape_[split_id_]; }

void Split::print_info() {
  printf("Split(");
  input_dim_trans_->print_info();
  printf(", (");
  for (int64_t i = 0, n = splitted_shape_.size(); i < n; ++i) {
    printf("%ld", splitted_shape_[i]);
    if (i < n - 1) {
      printf(",");
    }
  }
  printf("), %ld)", split_id_);
}

DimTrans* make_flatten(const std::vector<DimTrans*>& dims) {
  DimTrans* ptr = nullptr;
  if (dims.size() == 0) {
    ptr = new Singleton();
  } else if (dims.size() == 1) {
    ptr = dims[0];
  } else {
    ptr = new Flatten(dims);
  }
  return ptr;
}

DimTrans* make_split(DimTrans* dim,
                     const std::vector<int64_t>& shape,
                     int64_t id) {
  assert(shape.size() > 0);
  DimTrans* ptr = nullptr;
  if (shape.size() == 1) {
    assert(id == 0);
    ptr = dim;
  } else if (shape[id] == 1) {
    ptr = new Singleton();
  } else {
    // new shape that remove 1
    std::vector<int64_t> new_shape;
    // map between from idx in shape to new_shape
    std::vector<int64_t> idx_map(shape.size(), -1);
    for (int64_t i = 0, n = shape.size(); i < n; ++i) {
      if (shape[id] != 1) {
        idx_map[i] = new_shape.size();
        new_shape.emplace_back(shape[i]);
      }
    }
    ptr = new Split(dim, new_shape, idx_map[id]);
  }
  return ptr;
}

void CleanUp() {
  for (int64_t i = 0, n = all_dim_trans.size(); i < n; i++) {
    if (all_dim_trans[i]) {
      delete all_dim_trans[i];
      all_dim_trans[i] = nullptr;
    }
  }
  std::vector<DimTrans*>().swap(all_dim_trans);
}

// Given a `dim_trans` of an output axis, get the size and
// the input axis whose dim mapping should be propogated to
// the output axis. If the returned input axis is none, the
// output axis's dim mapping should be set to -1 (replicated).
// For an axis that is flattened from input axes, return the
// leftmost flattened input axis. For the split transformation,
// only the leftmost split axis in output will return its input.
std::pair<int64_t, DimTrans*> GetDimTransSize(
    DimTrans* dim_trans,
    std::vector<std::vector<bool>>* shardable,
    std::set<int64_t>* seen_dims,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& mesh_shape,
    const std::vector<int64_t>& input_dims_mapping,
    const std::set<int64_t>& sharded_input_dims) {
  DimTrans::Type type = dim_trans->type();
  DimTrans* ret_dim_trans = nullptr;
  int64_t ret_size = -1;

  if (type == DimTrans::Type::INPUTDIM) {
    InputDim* inputdim = dynamic_cast<InputDim*>(dim_trans);
    int64_t dim = inputdim->input_dim();
    seen_dims->insert(dim);

    if (sharded_input_dims.count(dim) > 0) {
      ret_dim_trans = dim_trans;
    }
    ret_size = input_shape[dim];
  } else if (type == DimTrans::Type::FLATTEN) {
    Flatten* flatten = dynamic_cast<Flatten*>(dim_trans);
    const std::vector<DimTrans*>& inputs = flatten->inputs();
    int64_t nmesh = (*shardable)[0].size();
    std::vector<int64_t> input_sizes;
    for (int64_t i = 1, n = inputs.size(); i < n; i++) {
      DimTrans* input = inputs[i];
      if (input->type() == DimTrans::Type::INPUTDIM) {
        (*shardable)[i].assign(nmesh, false);
      }

      std::pair<int64_t, DimTrans*> dim_size =
          GetDimTransSize(input,
                          shardable,
                          seen_dims,
                          input_shape,
                          mesh_shape,
                          input_dims_mapping,
                          sharded_input_dims);
      input_sizes.emplace_back(dim_size.first);
      ret_size = std::accumulate(input_sizes.begin(),
                                 input_sizes.end(),
                                 1,
                                 std::multiplies<int64_t>());
    }

    DimTrans* dim0 = inputs[0];
    if (dim0->type() == DimTrans::Type::INPUTDIM) {
      InputDim* inputdim = dynamic_cast<InputDim*>(dim0);
      if (sharded_input_dims.count(inputdim->input_dim()) > 0) {
        ret_dim_trans = dim0;
      }
    }
  } else if (type == DimTrans::Type::SPLIT) {
    Split* split = dynamic_cast<Split*>(dim_trans);
    std::pair<int64_t, DimTrans*> dim_size =
        GetDimTransSize(split->input(),
                        shardable,
                        seen_dims,
                        input_shape,
                        mesh_shape,
                        input_dims_mapping,
                        sharded_input_dims);
    ret_size = split->local_split_shape();

    if (split->split_id() == 0) {
      if (dim_size.second != nullptr) {
        PADDLE_ENFORCE_EQ(dim_size.second->type(), DimTrans::Type::INPUTDIM);
        InputDim* inputdim = dynamic_cast<InputDim*>(dim_size.second);
        int64_t nmesh = mesh_shape.size();
        int64_t dim = inputdim->input_dim();

        // Check whether the sharded dim can be sharded on
        // each mesh dimension. The input dimension should be
        // divisible by the mesh size that it is sharded on
        for (int64_t imesh = 0; imesh < nmesh; imesh++) {
          (*shardable)[dim][imesh] = (ret_size % mesh_shape[imesh] == 0);
        }
      }
      ret_dim_trans = dim_size.second;
    }
  } else if (type == DimTrans::Type::SINGLETON) {
    ret_size = 1;
    ret_dim_trans = nullptr;
  }
  return std::make_pair(ret_size, ret_dim_trans);
}

void GetUsedInputDim(DimTrans* dim_trans, std::set<int64_t>* seen_dims) {
  if (dim_trans->type() == DimTrans::Type::INPUTDIM) {
    InputDim* input = dynamic_cast<InputDim*>(dim_trans);
    seen_dims->insert(input->input_dim());
  } else if (dim_trans->type() == DimTrans::Type::FLATTEN) {
    Flatten* flatten = dynamic_cast<Flatten*>(dim_trans);
    for (DimTrans* trans : flatten->inputs()) {
      GetUsedInputDim(trans, seen_dims);
    }
  } else if (dim_trans->type() == DimTrans::Type::SPLIT) {
    Split* split = dynamic_cast<Split*>(dim_trans);
    GetUsedInputDim(split->input(), seen_dims);
  } else {
    return;
  }
}

std::vector<std::vector<int64_t>> InferFromDimTrans(
    const DistTensorSpec& input_spec, const std::vector<DimTrans*>& dim_trans) {
  const std::vector<int64_t>& input_shape = input_spec.shape();
  const std::vector<int64_t>& input_dims_mapping = input_spec.dims_mapping();
  const ProcessMesh& mesh = input_spec.dist_attr().process_mesh();
  const std::vector<int64_t>& mesh_shape = mesh.shape();

  std::set<int64_t> sharded_input_dims;
  for (int64_t i = 0, n = input_dims_mapping.size(); i < n; ++i) {
    if (input_dims_mapping[i] > -1) {
      sharded_input_dims.insert(i);
    }
  }
  int64_t ndim = input_shape.size();
  int64_t nmesh = mesh_shape.size();
  std::vector<std::vector<bool>> shardable(ndim,
                                           std::vector<bool>(nmesh, true));

  std::set<int64_t> seen_input_dims;
  for (DimTrans* trans : dim_trans) {
    GetUsedInputDim(trans, &seen_input_dims);
  }

  for (int64_t idim = 0; idim < ndim; idim++) {
    bool seen = seen_input_dims.count(idim);
    if (!seen) {
      shardable[idim].assign(nmesh, seen);
    }
  }

  // get the size of each output dimension, and get the
  // map from sharded input dimensions to output dimensions.
  std::vector<int64_t> dim_map_src2tgt(ndim, -1);
  std::vector<int64_t> out_shape(dim_trans.size());
  for (int64_t i = 0, n = dim_trans.size(); i < n; i++) {
    std::pair<int64_t, DimTrans*> dim_size =
        GetDimTransSize(dim_trans[i],
                        &shardable,
                        &seen_input_dims,
                        input_shape,
                        mesh_shape,
                        input_dims_mapping,
                        sharded_input_dims);
    out_shape[i] = dim_size.first;
    if (dim_size.second != nullptr &&
        dim_size.second->type() == DimTrans::Type::INPUTDIM) {
      InputDim* inputdim = dynamic_cast<InputDim*>(dim_size.second);
      dim_map_src2tgt[inputdim->input_dim()] = i;
    }
  }

  // if one input dimension is sharded on a
  // unshardable mesh we need to reshard the input.
  bool need_reshard = false;
  for (int64_t i = 0; i < ndim; i++) {
    int64_t mesh_dim = input_dims_mapping[i];
    if (input_dims_mapping[i] > -1 && !shardable[i][mesh_dim]) {
      need_reshard = true;
      break;
    }
  }

  std::vector<int64_t> out_dims_mapping(dim_trans.size(), -1);
  std::vector<int64_t> new_input_dims_mapping(input_dims_mapping);
  if (!need_reshard) {
    for (int64_t i = 0; i < ndim; i++) {
      if (input_dims_mapping[i] > -1 && dim_map_src2tgt[i] > -1) {
        out_dims_mapping[dim_map_src2tgt[i]] = input_dims_mapping[i];
      }
    }
  } else {
    // set the unshardable input dimension to be replicate
    for (int64_t i = 0; i < ndim; i++) {
      int64_t mesh_dim = input_dims_mapping[i];
      if (mesh_dim > -1 && shardable[i][mesh_dim] && dim_map_src2tgt[i] > -1) {
        out_dims_mapping[dim_map_src2tgt[i]] = input_dims_mapping[i];
      } else {
        new_input_dims_mapping[i] = -1;
      }
    }
  }

  return {new_input_dims_mapping, out_dims_mapping};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
