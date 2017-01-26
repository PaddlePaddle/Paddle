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

#include "Function.h"

namespace paddle {

template <>
size_t FuncConfig::get<size_t>(const std::string& key) const {
  auto it = valueMap_.find(key);
  CHECK(it != valueMap_.end()) << "Cannot find value: '" << key << "'";
  return it->second.s;
}

template <>
real FuncConfig::get<real>(const std::string& key) const {
  auto it = valueMap_.find(key);
  CHECK(it != valueMap_.end()) << "Cannot find value: '" << key << "'";
  return it->second.r;
}

template <>
int FuncConfig::get<int>(const std::string& key) const {
  auto it = valueMap_.find(key);
  CHECK(it != valueMap_.end()) << "Cannot find value: '" << key << "'";
  return it->second.i;
}

template <>
bool FuncConfig::get<bool>(const std::string& key) const {
  auto it = valueMap_.find(key);
  CHECK(it != valueMap_.end()) << "Cannot find value: '" << key << "'";
  return it->second.b;
}

template <>
FuncConfig& FuncConfig::set<size_t>(const std::string& key, size_t v) {
  CHECK_EQ(static_cast<int>(valueMap_.count(key)), 0) << "Duplicated value: "
                                                      << key;
  valueMap_[key].s = v;
  return *this;
}

template <>
FuncConfig& FuncConfig::set<real>(const std::string& key, real v) {
  CHECK_EQ(static_cast<int>(valueMap_.count(key)), 0) << "Duplicated value: "
                                                      << key;
  valueMap_[key].r = v;
  return *this;
}

template <>
FuncConfig& FuncConfig::set<int>(const std::string& key, int v) {
  CHECK_EQ(static_cast<int>(valueMap_.count(key)), 0) << "Duplicated value: "
                                                      << key;
  valueMap_[key].i = v;
  return *this;
}

template <>
FuncConfig& FuncConfig::set<bool>(const std::string& key, bool v) {
  CHECK_EQ(static_cast<int>(valueMap_.count(key)), 0) << "Duplicated value: "
                                                      << key;
  valueMap_[key].b = v;
  return *this;
}

void BufferArgs::addArg(const Matrix& arg,
                        const TensorShape& shape,
                        ArgType argType) {
  _args_.push_back(new BufferArg(arg, shape, argType));
  addArg(*_args_.back());
}

void BufferArgs::addArg(const CpuSparseMatrix& arg, ArgType argType) {
  _args_.push_back(new SparseMatrixArg(arg, argType));
  addArg(*_args_.back());
}

void BufferArgs::addArg(const GpuSparseMatrix& arg, ArgType argType) {
  _args_.push_back(new SparseMatrixArg(arg, argType));
  addArg(*_args_.back());
}

void BufferArgs::addArg(const Matrix& matrix,
                        const IVector& vector,
                        ArgType argType) {
  _args_.push_back(new SequenceArg(matrix, vector, argType));
  addArg(*_args_.back());
}

ClassRegistrar<FunctionBase> FunctionBase::funcRegistrar_;

}  // namespace paddle
