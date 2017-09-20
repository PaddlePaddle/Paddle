/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <map>
#include <vector>
#include "paddle/framework/framework.pb.h"

namespace paddle {
namespace framework {

/*
 * SymbolTable is the container of OpDesc and VarDesc.
 */
class SymbolTable {
 public:
  SymbolTable() {}
  explicit SymbolTable(SymbolTable* parent) : parent_(parent) {}

  /*
   * Create a new OpDesc.
   */
  const OpDesc* NewOp();

  /*
   * Create a new VarDesc.
   */
  const VarDesc* NewVar(const std::string& name);

  /*
   * Find a VarDesc by name, if recursive true, find parent's SymbolTable
   * recursively. This interface is proposed to support InferShape, find
   * protobuf messages or variables and operators, pass pointers into
   * InferShape.
   *
   * NOTE(superjom) maybe some C++ classes such as VarDescBuilder and
   * OpDescBuilder should be proposed and embedded into pybind to enable python
   * operate on C++ pointers.
   */
  const VarDesc* FindVar(const std::string& name, bool recursive = true) const;

  /*
   * Find a OpDesc by idx.
   */
  const OpDesc* FindOp(size_t idx) const;

  /*
   * Create a BlockDesc based on the registered VarDesc and OpDesc.
   */
  BlockDesc Compile() const;

 private:
  SymbolTable* parent_{nullptr};
  std::vector<OpDesc> ops_;
  std::map<std::string, VarDesc> vars_;
};

}  // namespace framework
}  // namespace paddle
