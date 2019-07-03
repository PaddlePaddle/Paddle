// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/gen_code/paddle_infer.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/op_lite.h"

namespace paddle {
namespace gencode {

void Tensor::Resize(const Tensor::ddim_t &shape) {
  CHECK(raw_mutable_tensor_);
  auto *tensor = static_cast<lite::Tensor *>(raw_mutable_tensor_);
  tensor->Resize(shape);
}

std::vector<int64_t> Tensor::shape() const {
  CHECK(raw_tensor_);
  auto *tensor = static_cast<const lite::Tensor *>(raw_tensor_);
  return tensor->dims().Vectorize();
}

#define FOR_EACH_TYPE(HANDLE) \
  HANDLE(int);                \
  HANDLE(float);              \
  HANDLE(int8_t);             \
  HANDLE(int64_t);

#define IMPL_DATA(T)                                                     \
  template <>                                                            \
  const T *Tensor::data<T>() const {                                     \
    CHECK(raw_tensor_);                                                  \
    const auto *tensor = static_cast<const lite::Tensor *>(raw_tensor_); \
    return tensor->data<T>();                                            \
  }
FOR_EACH_TYPE(IMPL_DATA);
#undef IMPL_DATA

#define IMPL_MUTABLE_DATA(T)                                         \
  template <>                                                        \
  T *Tensor::mutable_data<T>() {                                     \
    CHECK(raw_mutable_tensor_);                                      \
    auto *tensor = static_cast<lite::Tensor *>(raw_mutable_tensor_); \
    return tensor->mutable_data<T>();                                \
  }
FOR_EACH_TYPE(IMPL_MUTABLE_DATA);
#undef IMPL_MUTABLE_DATA

PaddlePredictor::PaddlePredictor() {
  raw_ops_ = new std::vector<std::shared_ptr<lite::OpLite>>;
  raw_kernels_ = new std::vector<std::unique_ptr<lite::KernelBase>>;
  raw_scope_ = new lite::Scope;
  raw_exe_scope_ = &(static_cast<lite::Scope *>(raw_scope_)->NewScope());
}

std::unique_ptr<Tensor> PaddlePredictor::GetTensor(
    const std::string &id) const {
  auto *exe_scope = static_cast<lite::Scope *>(raw_exe_scope_);
  const auto *var = exe_scope->FindVar(id);
  const auto &tensor = var->Get<lite::Tensor>();
  return std::unique_ptr<Tensor>(new Tensor(&tensor, nullptr));
}

std::unique_ptr<Tensor> PaddlePredictor::GetMutableTensor(
    const std::string &id) {
  auto *exe_scope = static_cast<lite::Scope *>(raw_exe_scope_);
  auto *var = exe_scope->FindVar(id);
  auto *tensor = var->GetMutable<lite::Tensor>();
  return std::unique_ptr<Tensor>(new Tensor(nullptr, tensor));
}

#define CAST_OPS \
  auto *ops =    \
      static_cast<std::vector<std::shared_ptr<lite::OpLite>> *>(raw_ops_);
#define CAST_KERNELS                                                 \
  auto *kernels =                                                    \
      static_cast<std::vector<std::unique_ptr<lite::KernelBase>> *>( \
          raw_kernels_);
#define CAST_SCOPE auto *scope = static_cast<lite::Scope *>(raw_scope_);

PaddlePredictor::~PaddlePredictor() {
  CAST_OPS
  CAST_KERNELS
  CAST_SCOPE

  if (ops) {
    delete ops;
  }
  if (kernels) {
    delete kernels;
  }
  if (scope) {
    delete scope;
  }
}

void PaddlePredictor::Run() {
  CAST_OPS
  CAST_KERNELS

  CHECK(ops);
  CHECK(kernels);
  CHECK_EQ(ops->size(), kernels->size());

  for (size_t i = 0; i < ops->size(); i++) {
    LOG(INFO) << "Running the " << i << "-th operator";
    ops->at(i)->InferShape();
    kernels->at(i)->Launch();
  }
}

std::unique_ptr<Tensor> PaddlePredictor::GetInput(size_t offset) {
  auto *exec_scope = static_cast<lite::Scope *>(raw_exe_scope_);
  auto *_feed_list = exec_scope->FindVar("feed");
  CHECK(_feed_list) << "no feed variable in exec_scope";
  auto *feed_list = _feed_list->GetMutable<std::vector<lite::Tensor>>();
  if (offset >= feed_list->size()) {
    feed_list->resize(offset + 1);
  }

  return std::unique_ptr<Tensor>(new Tensor(nullptr, &feed_list->at(offset)));
}

std::unique_ptr<Tensor> PaddlePredictor::GetOutput(size_t offset) {
  auto *exec_scope = static_cast<lite::Scope *>(raw_exe_scope_);
  auto *_fetch_list = exec_scope->FindVar("fetch");
  CHECK(_fetch_list) << "no fatch variable in exec_scope";
  auto &fetch_list = *_fetch_list->GetMutable<std::vector<lite::Tensor>>();
  CHECK_LT(offset, fetch_list.size()) << "offset " << offset << " overflow";
  return std::unique_ptr<Tensor>(new Tensor(&fetch_list.at(offset), nullptr));
}

}  // namespace gencode
}  // namespace paddle
