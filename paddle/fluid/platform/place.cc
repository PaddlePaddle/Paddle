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

#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {

static Place *the_current_place = new CPUPlace();

bool operator==(const Place &p1, const Place &p2) {
  return is_same_place(p1, p2);
}

bool operator!=(const Place &p1, const Place &p2) {
  return !is_same_place(p1, p2);
}

Place *clone_place(const Place &p) {
  Place *r = nullptr;
  if (is_cpu_place(p))
    r = new CPUPlace();
  else if (is_gpu_place(p))
    r = new CUDAPlace(dynamic_cast<const CUDAPlace &>(p).device);
  else if (is_cuda_pinned_place(p))
    r = new CUDAPinnedPlace();
  return r;
}

void set_place(const Place &p) {
  delete the_current_place;
  the_current_place = clone_place(p);
}

const Place &get_place() { return *the_current_place; }

const CUDAPlace default_gpu() { return CUDAPlace(0); }
const CPUPlace default_cpu() { return CPUPlace(); }
const CUDAPinnedPlace default_cuda_pinned() { return CUDAPinnedPlace(); }

bool is_gpu_place(const Place &p) {
  return dynamic_cast<const CUDAPlace *>(&p) != nullptr;
}

bool is_cpu_place(const Place &p) {
  return dynamic_cast<const CPUPlace *>(&p) != nullptr;
}

bool is_cuda_pinned_place(const Place &p) {
  return dynamic_cast<const CUDAPinnedPlace *>(&p) != nullptr;
}

int which_place(const Place &p) {
  int i = -1;
  if (is_cpu_place(p))
    i = 0;
  else if (is_gpu_place(p))
    i = 1;
  else if (is_cuda_pinned_place(p))
    i = 2;
  return i;
}

bool places_are_same_class(const Place &p1, const Place &p2) {
  return which_place(p1) == which_place(p2);
}

bool is_same_place(const Place &p1, const Place &p2) {
  if (places_are_same_class(p1, p2)) {
    return is_gpu_place(p1) ? dynamic_cast<const CUDAPlace &>(p1).device ==
                                  dynamic_cast<const CUDAPlace &>(p2).device
                            : true;
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, const Place &p) {
  if (is_cpu_place(p))
    os << "CPUPlace";
  else if (is_gpu_place(p))
    os << "CUDAPlace(" << dynamic_cast<const CUDAPlace &>(p).device << ")";
  else if (is_cuda_pinned_place(p))
    os << "CUDAPinnedPlace";
  return os;
}

}  // namespace platform
}  // namespace paddle
