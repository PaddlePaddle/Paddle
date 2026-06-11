# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sleef library configuration
# Sleef provides vectorized math functions with high precision and performance

option(
  WITH_SLEEF
  "Compile PaddlePaddle with Sleef library for high precision math functions"
  OFF)

if(WITH_SLEEF)
  # Always build Sleef from the in-tree submodule (third_party/sleef) to
  # guarantee consistent feature set (e.g. AVX/AVX2 256-bit symbols like
  # Sleef_sinf8_u35) across local and CI environments. System-installed
  # sleef is intentionally not used because its build options are
  # uncontrolled and may miss required dispatch variants.
  include(external/sleef)

  if(TARGET extern_sleef)
    add_definitions(-DPADDLE_WITH_SLEEF)
    message(STATUS "Compile with Sleef support")

    # Add precision control option
    set(PADDLE_SLEEF_POW_PRECISION
        10
        CACHE
          STRING
          "Sleef pow function precision: 10 (1.0 ULP), 15 (1.5 ULP), 35 (3.5 ULP)"
    )
    set_property(CACHE PADDLE_SLEEF_POW_PRECISION PROPERTY STRINGS 10 15 35)

    # Validate precision option
    if(NOT PADDLE_SLEEF_POW_PRECISION MATCHES "^(10|15|35)$")
      message(FATAL_ERROR "PADDLE_SLEEF_POW_PRECISION must be 10, 15, or 35")
    endif()

    add_definitions(-DPADDLE_SLEEF_POW_PRECISION=${PADDLE_SLEEF_POW_PRECISION})
    message(
      STATUS "Sleef pow precision set to: ${PADDLE_SLEEF_POW_PRECISION} ULP")
  else()
    message(WARNING "Sleef not found, disabling Sleef support")
    set(WITH_SLEEF OFF)
  endif()
endif()
