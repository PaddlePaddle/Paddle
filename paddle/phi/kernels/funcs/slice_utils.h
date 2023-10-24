/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <glog/logging.h>
#include <paddle/phi/core/ddim.h>
#include <string>
#include <vector>
#include "paddle/phi/core/flags.h"

PHI_DECLARE_bool(set_to_1d);

namespace phi {

namespace funcs {

template <typename T = int64_t>
inline void CheckAndUpdateSliceAttrs(const DDim in_dims,
                                     const std::vector<T>& axes,
                                     std::vector<T>* starts,
                                     std::vector<T>* ends,
                                     std::vector<int64_t>* steps = nullptr,
                                     std::vector<T>* infer_flags = nullptr) {
  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    PADDLE_ENFORCE_LT(
        axis,
        in_dims.size(),
        phi::errors::InvalidArgument(
            "The axis value should be less than the rank of input, "
            "but received axes[%d] = %d, rank of input is %d.",
            i,
            axis,
            in_dims.size()));

    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      continue;
    }

    T dim_value = in_dims[axis];

    if (dim_value > 0) {
      T step = steps == nullptr ? 1 : (*steps)[i];
      PADDLE_ENFORCE_NE(
          step,
          0,
          phi::errors::InvalidArgument(
              "Step should not be 0, but received step = %d.", step));

      T start = (*starts)[i] < 0 ? ((*starts)[i] + dim_value) : (*starts)[i];
      start = std::max(start, static_cast<T>(0));

      T end =
          0 < step && (*ends)[i] < 0 ? ((*ends)[i] + dim_value) : (*ends)[i];
      end = std::min(end, dim_value);

      if (step > 0) {
        start = std::min(start, dim_value);
        end = std::max(end, static_cast<T>(0));
        PADDLE_ENFORCE_GE(
            end,
            start,
            phi::errors::InvalidArgument(
                "When step > 0, end should be greater than start, but "
                "received end = %d, start = %d.",
                end,
                start));
      } else {
        // NOTE(liym27): When step < 0, start should less and equal to
        // dim_value-1
        // "end is -1" means contain the 0-th element of this axis.
        start = std::min(start, dim_value - 1);
        if (end < -1) {
          end += dim_value;
        }
        end = std::max(end, static_cast<T>(-1));
        PADDLE_ENFORCE_GE(
            start,
            end,
            phi::errors::InvalidArgument(
                "When step < 0, start should be greater than end, but "
                "received start = %d, end = %d.",
                start,
                end));
      }

      (*starts)[i] = start;
      (*ends)[i] = end;
    } else if (dim_value == 0) {
      (*starts)[i] = 0;
      (*ends)[i] = 0;
    }
  }
}

template <typename T = int64_t>
inline void UpdateSliceAttrs(const DDim in_dims,
                             const std::vector<T>& axes,
                             std::vector<T>* starts,
                             std::vector<T>* ends,
                             std::vector<int64_t>* steps = nullptr,
                             std::vector<T>* infer_flags = nullptr) {
  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      continue;
    }
    T dim_value = in_dims[axis];
    if (dim_value > 0) {
      T step = steps == nullptr ? 1 : (*steps)[i];
      T start = (*starts)[i] < 0 ? ((*starts)[i] + dim_value) : (*starts)[i];
      start = std::max(start, static_cast<T>(0));
      T end =
          0 < step && (*ends)[i] < 0 ? ((*ends)[i] + dim_value) : (*ends)[i];
      end = std::min(end, dim_value);

      if (step > 0) {
        start = std::min(start, dim_value);
        end = std::max(end, static_cast<T>(0));
      } else {
        // NOTE: When step < 0, start should less and equal to
        // dim_value-1
        // "end is -1" means contain the 0-th element of this axis.
        start = std::min(start, dim_value - 1);
        if (end < -1) {
          end += dim_value;
        }
        end = std::max(end, static_cast<T>(-1));
      }
      (*starts)[i] = start;
      (*ends)[i] = end;
    } else if (dim_value == 0) {
      (*starts)[i] = 0;
      (*ends)[i] = 0;
    }
  }
}

template <typename T = int64_t>
inline phi::DDim GetSliceDims(const phi::DDim in_dims,
                              const std::vector<T>& axes,
                              const std::vector<T>& starts,
                              const std::vector<T>& ends,
                              std::vector<T>* steps = nullptr,
                              std::vector<T>* infer_flags = nullptr) {
  phi::DDim slice_dims(in_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      slice_dims[axis] = -1;
      continue;
    }

    if (in_dims[axis] == -1) {
      continue;
    }

    T start = starts[i];
    T end = ends[i];
    T step = steps == nullptr ? 1 : (*steps)[i];

    if (step > 0) {
      slice_dims[axis] = (end - start + step - 1) / step;
    } else {
      slice_dims[axis] = (end - start + step + 1) / step;
    }
  }
  return slice_dims;
}

template <typename T = int64_t>
inline DDim GetDecreasedDims(const DDim slice_dims,
                             const std::vector<T>& decrease_axes,
                             std::vector<T>* infer_flags = nullptr) {
  DDim decreased_dims(slice_dims);
  std::vector<uint8_t> decrease_flag(slice_dims.size(), 0);
  if (decrease_axes.size() > 0) {
    for (size_t i = 0; i < decrease_axes.size(); ++i) {
      T axis = decrease_axes[i];
      decrease_flag[axis] = 1;
      if (infer_flags && (*infer_flags)[i] != -1) {
        PADDLE_ENFORCE_EQ(decreased_dims[axis],
                          1,
                          phi::errors::InvalidArgument(
                              "Decrease dim should be 1, but now received %d",
                              decreased_dims[axis]));
      }
    }

    std::vector<T> new_shape;
    for (int i = 0; i < decreased_dims.size(); ++i) {
      if (decrease_flag[i] == 0) {
        new_shape.push_back(decreased_dims[i]);
      }
    }
    if (FLAGS_set_to_1d && new_shape.size() == 0) {
      // NOTE(zoooo0820): Hack procssing to 1-D, when axes decrease to 0-D in
      // slice. This will remove in release 2.6.
      new_shape.push_back(1);
    }
    decreased_dims = phi::make_ddim(new_shape);
  }
  return decreased_dims;
}

template <typename T = int64_t>
inline void CheckAndUpdateSparseSliceAttrs(const DDim in_dims,
                                           std::vector<T>* axes,
                                           std::vector<T>* starts,
                                           std::vector<T>* ends) {
  int64_t rank = int64_t(in_dims.size());
  for (auto& axis : *axes) {
    if (axis < 0) {
      axis = std::max(int64_t(0), axis + rank);
    }
  }

  PADDLE_ENFORCE_EQ(
      axes->size(),
      starts->size(),
      phi::errors::InvalidArgument(
          "The length of axes (%d) and length of starts (%d) should be same.",
          axes->size(),
          starts->size()));
  PADDLE_ENFORCE_EQ(
      axes->size(),
      ends->size(),
      phi::errors::InvalidArgument(
          "The length of axes (%d) and length of ends (%d) should be same.",
          axes->size(),
          ends->size()));

  CheckAndUpdateSliceAttrs<T>(in_dims, *axes, starts, ends);
}

inline void ConstructNewSliceAttrs(const phi::DDim& x_dims,
                                   const std::vector<int64_t>& axes,
                                   const std::vector<int64_t>& starts,
                                   const std::vector<int64_t>& ends,
                                   std::vector<int64_t>* new_axes,
                                   std::vector<int64_t>* new_starts,
                                   std::vector<int64_t>* new_ends) {
  for (int64_t i = 0; i < x_dims.size(); ++i) {
    int pos = -1;
    for (int j = 0; j < static_cast<int>(axes.size()); ++j) {
      if (axes[j] == i) {
        pos = j;
        break;
      }
    }
    if (pos == -1) {
      (*new_axes)[i] = i;
      (*new_starts)[i] = 0;
      (*new_ends)[i] = x_dims[i];
    } else {
      (*new_axes)[i] = axes[pos];
      (*new_starts)[i] = starts[pos];
      (*new_ends)[i] = ends[pos];
    }
  }
}

}  // namespace funcs
}  // namespace phi
