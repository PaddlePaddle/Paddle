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

using phi::distributed::auto_parallel::str_join;

#define EXTRACT_SHAPE_AND_DIST_ATTR(x)                                 \
  auto x##_shape = phi::vectorize(x.dims());                           \
  int x##_ndim = x##_shape.size();                                     \
  const auto& x##_dist_attr_src = x.dist_attr();                       \
  const auto& x##_dims_mapping_src = x##_dist_attr_src.dims_mapping(); \
  PADDLE_ENFORCE_EQ(x##_ndim,                                          \
                    x##_dims_mapping_src.size(),                       \
                    common::errors::InvalidArgument(                   \
                        "[%d] [%d] The Tensor [%d]'s rank [%d] and "   \
                        "dims_mapping size [%d] are not matched.",     \
                        __FILE__,                                      \
                        __LINE__,                                      \
                        #x,                                            \
                        x##_ndim,                                      \
                        x##_dims_mapping_src.size()))

#define EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x)                   \
  EXTRACT_SHAPE_AND_DIST_ATTR(x);                                    \
  PADDLE_ENFORCE_EQ(x##_ndim,                                        \
                    x##_dims_mapping_src.size(),                     \
                    common::errors::InvalidArgument(                 \
                        "[%d] [%d] The Tensor [%d]'s rank [%d] and " \
                        "dims_mapping size [%d] are not matched.",   \
                        __FILE__,                                    \
                        __LINE__,                                    \
                        #x,                                          \
                        x##_ndim,                                    \
                        x##_dims_mapping_src.size()))

#define LOG_SPMD_INPUT(name)                                                  \
  do {                                                                        \
    VLOG(4) << #name;                                                         \
    VLOG(4) << "shape: [" << str_join(name##_shape) << "] "                   \
            << "src_dist_attr: [" << name##_dist_attr_src.to_string() << "] " \
            << "dst_dist_attr: [" << name##_dist_attr_dst.to_string() << "]"; \
  } while (0)

#define LOG_SPMD_OUTPUT(name)                             \
  do {                                                    \
    VLOG(4) << #name;                                     \
    VLOG(4) << "dist_attr: [" << name.to_string() << "]"; \
  } while (0)
