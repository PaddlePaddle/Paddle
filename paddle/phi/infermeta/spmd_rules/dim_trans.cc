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

#include "paddle/phi/infermeta/spmd_rules/dim_trans.h"
#include <assert.h>
#include <cstdio>
#include <numeric>
#include <set>
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {

static std::vector<DimTrans*> all_dim_trans;

DimTrans::DimTrans(Type type) : type_(type) {}

DimTrans::~DimTrans() {}

DimTrans::Type DimTrans::type() const { return type_; }

void DimTrans::set_type(Type type) { type_ = type; }

std::string DimTrans::to_string() { return std::string(""); }

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

std::string InputDim::to_string() {
  return ("InputDim(" + std::to_string(input_dim_) + ")");
}

Singleton::Singleton() : DimTrans(DimTrans::Type::SINGLETON) {
  all_dim_trans.emplace_back(this);
}

std::string Singleton::to_string() { return "Singleton()"; }

Flatten::Flatten() : DimTrans(DimTrans::Type::FLATTEN) {
  all_dim_trans.emplace_back(this);
}

Flatten::Flatten(const std::vector<DimTrans*>& dims)
    : DimTrans(DimTrans::Type::FLATTEN) {
  input_dims_ = dims;
  all_dim_trans.emplace_back(this);
}

Flatten::~Flatten() {  // NOLINT
  input_dims_.assign(input_dims_.size(), nullptr);
  std::vector<DimTrans*>().swap(input_dims_);
}

const std::vector<DimTrans*>& Flatten::inputs() const { return input_dims_; }

void Flatten::set_inputs(const std::vector<DimTrans*>& dims) {
  input_dims_.assign(dims.begin(), dims.end());
}

std::string Flatten::to_string() {
  std::string ret_str("Flatten(");
  for (int i = 0, n = static_cast<int>(input_dims_.size()); i < n; ++i) {
    ret_str += input_dims_[i]->to_string();
    if (i < n - 1) {
      ret_str += ",";
    }
  }
  return ret_str + ")";
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

int64_t Split::local_splitted_shape_value() {
  return splitted_shape_[split_id_];
}

std::string Split::to_string() {
  std::string ret_str("Split(");
  ret_str += input_dim_trans_->to_string() + ", (";
  for (int i = 0, n = static_cast<int>(splitted_shape_.size()); i < n; ++i) {
    ret_str += std::to_string(splitted_shape_[i]);
    if (i < n - 1) {
      ret_str += ",";
    }
  }
  return ret_str + "), " + std::to_string(split_id_) + ")";
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
    for (int i = 0, n = static_cast<int>(shape.size()); i < n; ++i) {
      if (shape[id] != 1) {
        idx_map[i] = static_cast<int64_t>(new_shape.size());
        new_shape.emplace_back(shape[i]);
      }
    }
    ptr = new Split(dim, new_shape, idx_map[id]);
  }
  return ptr;
}

void CleanUp() {
  int n = static_cast<int>(all_dim_trans.size());
  for (int i = 0; i < n; i++) {
    if (all_dim_trans[i]) {
      delete all_dim_trans[i];
      all_dim_trans[i] = nullptr;
    }
  }
  std::vector<DimTrans*>().swap(all_dim_trans);
}

// Given a `dim_trans` of an output axis, get the input axis
// whose dim mapping should be propogated to it.
// If the returned input axis is none, the output axis's
// dim mapping should be set to -1 (replicated). For an axis
// that is flattened from input axes, return the leftmost
// flattened input axis. For the split transformation,
// only the leftmost split axis in output will return its input.
DimTrans* GetDimTrans(DimTrans* dim_trans,
                      std::vector<std::vector<bool>>* shardable,
                      std::set<int64_t>* seen_dims,
                      const std::vector<int64_t>& input_shape,
                      const std::vector<int64_t>& mesh_shape,
                      const std::vector<int64_t>& input_dims_mapping,
                      const std::set<int64_t>& sharded_input_dims) {
  DimTrans::Type type = dim_trans->type();
  DimTrans* ret_dim_trans = nullptr;

  if (type == DimTrans::Type::INPUTDIM) {
    InputDim* inputdim = dynamic_cast<InputDim*>(dim_trans);
    int64_t dim = inputdim->input_dim();
    seen_dims->insert(dim);

    if (sharded_input_dims.count(dim) > 0) {
      ret_dim_trans = dim_trans;
    }
  } else if (type == DimTrans::Type::FLATTEN) {
    Flatten* flatten = dynamic_cast<Flatten*>(dim_trans);
    const std::vector<DimTrans*>& inputs = flatten->inputs();
    int64_t nmesh = (*shardable)[0].size();  // NOLINT
    for (int i = 1, n = static_cast<int>(inputs.size()); i < n; i++) {
      DimTrans* input = inputs[i];
      if (input->type() == DimTrans::Type::INPUTDIM) {
        InputDim* inputdim = dynamic_cast<InputDim*>(input);
        (*shardable)[inputdim->input_dim()].assign(nmesh, false);
      }

      GetDimTrans(input,
                  shardable,
                  seen_dims,
                  input_shape,
                  mesh_shape,
                  input_dims_mapping,
                  sharded_input_dims);
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
    DimTrans* dim = GetDimTrans(split->input(),
                                shardable,
                                seen_dims,
                                input_shape,
                                mesh_shape,
                                input_dims_mapping,
                                sharded_input_dims);
    int64_t ret_size = split->local_splitted_shape_value();

    if (split->split_id() == 0) {
      if (dim != nullptr) {
        PADDLE_ENFORCE_EQ(dim->type(),
                          DimTrans::Type::INPUTDIM,
                          phi::errors::InvalidArgument(
                              "The returned dim_trans must be INPUTDIM."));
        InputDim* inputdim = dynamic_cast<InputDim*>(dim);
        int64_t nmesh = static_cast<int64_t>(mesh_shape.size());
        int64_t input_axis = inputdim->input_dim();

        // Check whether the sharded dim can be sharded on
        // each mesh dimension. The dimension should be
        // divisible by the mesh size that it is sharded on
        for (int64_t imesh = 0; imesh < nmesh; imesh++) {
          (*shardable)[input_axis][imesh] = (ret_size % mesh_shape[imesh] == 0);
        }
      }
      ret_dim_trans = dim;
    }
  } else if (type == DimTrans::Type::SINGLETON) {
    ret_dim_trans = nullptr;
  }
  return ret_dim_trans;
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
    const DistMetaTensor& input, const std::vector<DimTrans*>& dim_trans) {
  std::vector<int64_t> input_shape = phi::vectorize(input.dims());
  const std::vector<int64_t>& input_dims_mapping =
      input.dist_attr().dims_mapping();
  const ProcessMesh& mesh = input.dist_attr().process_mesh();
  const std::vector<int64_t>& mesh_shape = mesh.shape();

  std::set<int64_t> sharded_input_dims;
  for (int64_t i = 0, n = static_cast<int64_t>(input_dims_mapping.size());
       i < n;
       ++i) {
    if (input_dims_mapping[i] > -1) {
      sharded_input_dims.insert(i);
    }
  }
  int64_t ndim = static_cast<int64_t>(input_shape.size());
  int64_t nmesh = static_cast<int64_t>(mesh_shape.size());
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

  // get the map from sharded input dimensions to output dimensions.
  std::vector<int64_t> dim_map_src2tgt(ndim, -1);
  for (int64_t i = 0, n = static_cast<int64_t>(dim_trans.size()); i < n; i++) {
    DimTrans* dim = GetDimTrans(dim_trans[i],
                                &shardable,
                                &seen_input_dims,
                                input_shape,
                                mesh_shape,
                                input_dims_mapping,
                                sharded_input_dims);
    if (dim != nullptr && dim->type() == DimTrans::Type::INPUTDIM) {
      InputDim* inputdim = dynamic_cast<InputDim*>(dim);
      dim_map_src2tgt[inputdim->input_dim()] = i;
    }
  }

  std::vector<int64_t> out_dims_mapping(dim_trans.size(), -1);
  std::vector<int64_t> new_input_dims_mapping(input_dims_mapping);

  // set output dims mapping with corresponding input dimensions.
  // if one input dimension is sharded on a unshardable mesh after
  // splitting, we need to make it replicated.
  for (int64_t i = 0; i < ndim; i++) {
    int64_t mesh_dim = input_dims_mapping[i];
    if (mesh_dim > -1 && shardable[i][mesh_dim] && dim_map_src2tgt[i] > -1) {
      out_dims_mapping[dim_map_src2tgt[i]] = input_dims_mapping[i];
    } else {
      new_input_dims_mapping[i] = -1;
    }
  }

  return {new_input_dims_mapping, out_dims_mapping};
}

}  // namespace distributed
}  // namespace phi
