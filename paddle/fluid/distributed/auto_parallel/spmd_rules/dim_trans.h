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

#include <iostream>
#include <vector>

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/dist_tensor_spec.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

// This is a base class to describe how each dimension in output tensor
// is transformed from input tensor's axes. The transformation includes
// Flatten, Split, etc. A vector<DimTrans*> whose size equals to the
// output tensor's rank can be used to describe how the output shape is
// transformed from the input shape. Each element in vector<DimTrans*>
// describes the transformation of one output axis. For example, when
// a reshape operator reshapes a tensor from the shape of (6, 12, 48)
// to (72, 6, 8), this transfromation can be described as:
// [Flatten(Dim(0), Dim(1)), Split(Dim(2), (6,8), 0), Split(Dim(2), (6,8), 1)]
// meaning that dim0 in output is flattened from dim0 and dim1 in input,
// dim1 and dim2 in output are obtained by splitting dim2 in input, the
// splitted shape is (6, 8), dim1 referes to the first shape value in (6, 8)
// and dim2 referes to the second shape value in (6, 8).
class DimTrans {
 public:
  enum class Type { INPUTDIM, SINGLETON, FLATTEN, SPLIT };

  DimTrans() = default;

  explicit DimTrans(Type type);

  virtual ~DimTrans();

  Type type() const;

  void set_type(Type type);

  virtual std::string to_string();

 private:
  Type type_;
};

// InputDim indicates that the output dimention
// is obtained directed from one input dimension.
class InputDim : public DimTrans {
 public:
  InputDim();

  explicit InputDim(int64_t dim);

  virtual ~InputDim();

  int64_t input_dim() const;

  void set_input_dim(int64_t dim);

  std::string to_string() override;

 private:
  int64_t input_dim_;
};

// Singleton indicates that the shape of the
// corresponding output dimension is 1
class Singleton : public DimTrans {
 public:
  Singleton();
  std::string to_string() override;
};

// Flatten indicates that the output dimension
// is obtained from flattening input dimensions.
class Flatten : public DimTrans {
 public:
  Flatten();

  explicit Flatten(const std::vector<DimTrans*>& dims);

  virtual ~Flatten();

  const std::vector<DimTrans*>& inputs() const;

  void set_inputs(const std::vector<DimTrans*>& dims);

  std::string to_string() override;

 private:
  std::vector<DimTrans*> input_dims_;
};

// Split indicates that the output dimension
// is obtained by splitting input dimension.
class Split : public DimTrans {
 public:
  Split();

  Split(DimTrans* dim, const std::vector<int64_t>& shape, int64_t id);

  virtual ~Split();

  DimTrans* input() const;

  void set_input(DimTrans* dim);

  int64_t split_id() const;

  // get the splitted shape of the split_id_ dimension
  int64_t local_split_shape();

  std::string to_string() override;

 private:
  DimTrans* input_dim_trans_;
  std::vector<int64_t> splitted_shape_;
  int64_t split_id_;
};

void CleanUp();

DimTrans* make_flatten(const std::vector<DimTrans*>& dims = {});

DimTrans* make_split(DimTrans* dim,
                     const std::vector<int64_t>& shape = {},
                     int64_t id = 0);

// Infer the dims mapping of the output tensor according to the transformation
// `dim_trans`. Returns the dims mapping of the input tensor (the input dims
// mapping may be changed for resharding) and output tensor. The inferring
// follows the rules:
// 1. For Singleton, i.e., the shape of this output axis is 1, its dim mapping
// is -1, indicating that the output axis is replicated.
// 2. For InputDim, i.e., the output axis is transformed directly from an input
// axis, set its dim mapping equals to the corresponding input axis.
// 3. For Flatten, i.e., the output axis is flattened from some input axes, it
// can be sharded only if the leftmost flattened axes is sharded.
// 4. For Split, i.e., the output axes is splited from a input axis, only the
// leftmost output split axis can be sharded when its shape can be divisible
// by the mesh dimension.
std::vector<std::vector<int64_t>> InferFromDimTrans(
    const DistTensorSpec& input_spec, const std::vector<DimTrans*>& dim_trans);

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
