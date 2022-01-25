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

#include <complex>
#include <mutex>
#include "paddle/pten/backends/dynload/lapack.h"
#include "paddle/pten/common/complex.h"

namespace paddle {
namespace platform {
namespace dynload {

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load lapack routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_LAPACK_WRAP(__name)                      \
  using DynLoad__##__name = pten::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_LAPACK_WRAP(__name) \
  DYNAMIC_LOAD_LAPACK_WRAP(__name)

#define LAPACK_ROUTINE_EACH(__macro) \
  __macro(dgetrf_);                  \
  __macro(sgetrf_);                  \
  __macro(zheevd_);                  \
  __macro(cheevd_);                  \
  __macro(dsyevd_);                  \
  __macro(ssyevd_);                  \
  __macro(dgeev_);                   \
  __macro(sgeev_);                   \
  __macro(zgeev_);                   \
  __macro(cgeev_);                   \
  __macro(dgels_);                   \
  __macro(sgels_);                   \
  __macro(dgelsd_);                  \
  __macro(sgelsd_);                  \
  __macro(dgelsy_);                  \
  __macro(sgelsy_);                  \
  __macro(dgelss_);                  \
  __macro(sgelss_);                  \
  __macro(zpotrs_);                  \
  __macro(cpotrs_);                  \
  __macro(dpotrs_);                  \
  __macro(spotrs_);

LAPACK_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_LAPACK_WRAP);

#undef DYNAMIC_LOAD_LAPACK_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
