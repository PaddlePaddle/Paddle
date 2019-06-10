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

#include "paddle/fluid/lite/core/op_lite.h"
#include <list>
#include <set>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

std::vector<std::unique_ptr<KernelBase>> OpLite::CreateKernels(
    const std::vector<Place> &places, const std::string &kernel_type) {
  std::vector<std::unique_ptr<KernelBase>> kernels;
  CHECK(!op_type_.empty()) << "op_type_ should be set first";

  auto pick_kernel = [&](const Place &place) {
    auto ks = KernelRegistry::Global().Create(
        (kernel_type.empty() ? op_type_ : kernel_type), place.target,
        place.precision, place.layout);
    for (auto &&it : ks) {
      AttachKernel(it.get());
      kernels.emplace_back(std::move(it));
    }
  };

  std::set<Place> place_set;
  for (auto place : places) {
    place_set.insert(place);
    // Pick kernels those support any Precision and any DataLayout
    place.precision = PRECISION(kAny);
    place_set.insert(place);
    place.layout = DATALAYOUT(kAny);
    place_set.insert(place);
  }

  std::set<TargetType> targets;
  for (auto place : place_set) {
    pick_kernel(place);
    targets.insert(place.target);
  }

  CHECK(!kernels.empty()) << "No kernel found for Op " << op_type_;
  VLOG(2) << "op " << op_type_ << " get " << kernels.size() << " kernels";
  return kernels;
}

bool OpLite::Run() {
  CHECK(kernel_);
  SyncInputEvents();

  kernel_->Launch();

  RecordOutputEvents();
  return true;
}

bool OpLite::Attach(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  // valid_places_.clear();
  CHECK(scope != nullptr);
  // CHECK(!op_info_.get());
  scope_ = scope;
  op_info_.reset(
      new OpInfo(opdesc));  // Force clean the out-of-date infomation.
  return AttachImpl(opdesc, scope);
}

const Tensor *OpLite::GetTensor(lite::Scope *scope,
                                const std::string &name) const {
  auto *var = scope->FindVar(name);
  CHECK(var) << "no variable called " << name << " found";
  return &var->Get<lite::Tensor>();
}

Tensor *OpLite::GetMutableTensor(lite::Scope *scope,
                                 const std::string &name) const {
  auto *var = scope->FindVar(name);
  CHECK(var) << "no variable called " << name << " found";
  return var->GetMutable<lite::Tensor>();
}

}  // namespace lite
}  // namespace paddle
