/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

struct Place {
  virtual ~Place() {}
};
  
struct CPUPlace : public Place {
  CPUPlace() {}
};

struct CUDAPlace : public Place {
  CUDAPlace() : CUDAPlace(0) {}
  explicit CUDAPlace(int d) : device(d) {}

  inline int GetDeviceId() const { return device; }
  int device;
};

struct CUDAPinnedPlace : public Place {
  CUDAPinnedPlace() {}
};

bool operator==(const Place&, const Place&);
bool operator!=(const Place&, const Place&);

using PlaceList = std::vector<Place>;  // TODO(yi): Remove it?

void set_place(const Place &);
const Place &get_place();

const CUDAPlace default_gpu();
const CPUPlace default_cpu();
const CUDAPinnedPlace default_cuda_pinned();

bool is_gpu_place(const Place &);
bool is_cpu_place(const Place &);
bool is_cuda_pinned_place(const Place &);
bool places_are_same_class(const Place &, const Place &);
bool is_same_place(const Place &, const Place &);

int which_place(const Place& p);
  
struct PlaceHash {
  std::size_t operator()(const Place &p) const {
    constexpr size_t num_dev_bits = 4;
    std::hash<int> ihash;
    size_t dev_id = 0;
    if (is_gpu_place(p)) {
      dev_id = dynamic_cast<const CUDAPlace&>(p).device;
    }
    return ihash(dev_id << num_dev_bits | which_place(p));
  }
};

std::ostream &operator<<(std::ostream &, const Place &);

}  // namespace platform
}  // namespace paddle
