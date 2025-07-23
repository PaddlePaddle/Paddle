// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/grid_sample_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/grid_sample_utils.h"

namespace phi {

template <typename T, typename IndexT>
static __forceinline__ __device__ void AtomicAdd(T* data,
                                                 IndexT h,
                                                 IndexT w,
                                                 IndexT sH,
                                                 IndexT sW,
                                                 IndexT H,
                                                 IndexT W,
                                                 T delta) {
  if (InBounds(h, w, H, W)) {
    phi::CudaAtomicAdd(data + h * sH + w * sW, delta);
  }
}

template <typename T, typename IndexT>
static __forceinline__ __device__ void AtomicAdd3D(T* data,
                                                   IndexT d,
                                                   IndexT h,
                                                   IndexT w,
                                                   IndexT sD,
                                                   IndexT sH,
                                                   IndexT sW,
                                                   IndexT D,
                                                   IndexT H,
                                                   IndexT W,
                                                   T delta) {
  if (InBounds3D(d, h, w, D, H, W)) {
    phi::CudaAtomicAdd(data + d * sD + h * sH + w * sW, delta);
  }
}

template <typename T, typename IndexT>
static __forceinline__ __device__ T
UnnormalizeWithMask(T coord, IndexT size, bool align_corners, T* grad_in) {
  if (align_corners) {
    *grad_in = static_cast<T>(size - 1) / 2;
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    *grad_in = static_cast<T>(size) / 2;
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename T, typename IndexT>
static __forceinline__ __device__ T ClipIndexesWithMask(T in,
                                                        IndexT clip_limit,
                                                        T* grad_in) {
  if (in <= static_cast<T>(0)) {
    *grad_in = static_cast<T>(0);
    return static_cast<T>(0);
  } else {
    T max = static_cast<T>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<T>(0);
      return max;
    } else {
      *grad_in = static_cast<T>(1);
      return in;
    }
  }
}

template <typename T, typename IndexT>
static __forceinline__ __device__ T
ReflectIndexesWithMask(T in, IndexT twice_low, IndexT twice_high, T* grad_in) {
  if (twice_low == twice_high) {
    *grad_in = static_cast<T>(0);
    return static_cast<T>(0);
  }
  IndexT grad_in_mult_;
  T min = static_cast<T>(twice_low) / 2;
  T span = static_cast<T>(twice_high - twice_low) / 2;
  in = in - min;
  if (in < static_cast<T>(0)) {
    grad_in_mult_ = -1;
    in = -in;
  } else {
    grad_in_mult_ = 1;
  }
  T extra = fmod(in, span);
  IndexT flips = static_cast<IndexT>(floor(in / span));
  if (flips % 2 == 0) {
    *grad_in = static_cast<T>(grad_in_mult_);
    return extra + min;
  } else {
    *grad_in = static_cast<T>(-grad_in_mult_);
    return span - extra + min;
  }
}

template <typename T, typename IndexT>
static __forceinline__ __device__ T
ComputePositionsWithMask(T coord,
                         IndexT size,
                         PaddingMode padding_mode,
                         bool align_corners,
                         T* grad_in) {
  T grad_clip, grad_refl;
  coord = UnnormalizeWithMask<T>(coord, size, align_corners, grad_in);
  if (padding_mode == PaddingMode::border) {
    coord = ClipIndexesWithMask(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == PaddingMode::reflect) {
    coord = align_corners ? ReflectIndexesWithMask<T, IndexT>(
                                coord, 0, 2 * (size - 1), &grad_refl)
                          : ReflectIndexesWithMask<T, IndexT>(
                                coord, -1, 2 * size - 1, &grad_refl);
    coord = ClipIndexesWithMask(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }
  return SafeDownGradeToIntRange(coord);
}

template <typename T, typename IndexT>
__global__ void GridSamplerCudaBackwardKernel(const IndexT nthreads,
                                              const T* grad_output,
                                              const T* input,
                                              const T* grid,
                                              IndexT n,
                                              IndexT out_c,
                                              IndexT out_h,
                                              IndexT out_w,
                                              IndexT in_h,
                                              IndexT in_w,
                                              T* grad_input,
                                              T* grad_grid,
                                              const Mode mode,
                                              const PaddingMode padding_mode,
                                              bool align_corners) {
  IndexT inp_sN = out_c * in_h * in_w;
  IndexT inp_sC = in_h * in_w;
  IndexT inp_sH = in_w;
  IndexT inp_sW = 1;
  IndexT grid_sN = out_h * out_w * 2;
  IndexT grid_sH = out_w * 2;
  IndexT grid_sW = 2;
  IndexT grid_sCoor = 1;

  IndexT gOut_sN = out_c * out_h * out_w;
  IndexT gOut_sC = out_h * out_w;
  IndexT gOut_sH = out_w;
  IndexT gOut_sW = 1;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const IndexT w = index % out_w;
    const IndexT h = (index / out_w) % out_h;
    const IndexT n = index / (out_h * out_w);
    const IndexT grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    T ix = grid[grid_offset];
    T iy = grid[grid_offset + grid_sCoor];

    T gix_mult, giy_mult;
    ix = ComputePositionsWithMask<T, IndexT>(
        ix, in_w, padding_mode, align_corners, &gix_mult);
    iy = ComputePositionsWithMask<T, IndexT>(
        iy, in_h, padding_mode, align_corners, &giy_mult);

    if (mode == Mode::bilinear) {
      IndexT ix_nw = static_cast<IndexT>(floor(ix));
      IndexT iy_nw = static_cast<IndexT>(floor(iy));
      IndexT ix_ne = ix_nw + 1;
      IndexT iy_ne = iy_nw;
      IndexT ix_sw = ix_nw;
      IndexT iy_sw = iy_nw + 1;
      IndexT ix_se = ix_nw + 1;
      IndexT iy_se = iy_nw + 1;

      T nw = (ix_se - ix) * (iy_se - iy);
      T ne = (ix - ix_sw) * (iy_sw - iy);
      T sw = (ix_ne - ix) * (iy - iy_ne);
      T se = (ix - ix_nw) * (iy - iy_nw);

      T gix = static_cast<T>(0), giy = static_cast<T>(0);
      IndexT gOut_offset = n * gOut_sN + h * gOut_sH + w * gOut_sW;
      T* gInp_ptr_NC = grad_input + n * inp_sN;
      IndexT inp_offset_NC = n * inp_sN;
      for (IndexT c = 0; c < out_c; ++c,
                  inp_offset_NC += inp_sC,
                  gInp_ptr_NC += inp_sC,
                  gOut_offset += gOut_sC) {
        T gOut = grad_output[gOut_offset];

        AtomicAdd(
            gInp_ptr_NC, iy_nw, ix_nw, inp_sH, inp_sW, in_h, in_w, nw * gOut);
        AtomicAdd(
            gInp_ptr_NC, iy_ne, ix_ne, inp_sH, inp_sW, in_h, in_w, ne * gOut);
        AtomicAdd(
            gInp_ptr_NC, iy_sw, ix_sw, inp_sH, inp_sW, in_h, in_w, sw * gOut);
        AtomicAdd(
            gInp_ptr_NC, iy_se, ix_se, inp_sH, inp_sW, in_h, in_w, se * gOut);

        if (InBounds(iy_nw, ix_nw, in_h, in_w)) {
          T nw_val = input[inp_offset_NC + iy_nw * inp_sH + ix_nw * inp_sW];
          gix -= nw_val * (iy_se - iy) * gOut;
          giy -= nw_val * (ix_se - ix) * gOut;
        }
        if (InBounds(iy_ne, ix_ne, in_h, in_w)) {
          T ne_val = input[inp_offset_NC + iy_ne * inp_sH + ix_ne * inp_sW];
          gix += ne_val * (iy_sw - iy) * gOut;
          giy -= ne_val * (ix - ix_sw) * gOut;
        }
        if (InBounds(iy_sw, ix_sw, in_h, in_w)) {
          T sw_val = input[inp_offset_NC + iy_sw * inp_sH + ix_sw * inp_sW];
          gix -= sw_val * (iy - iy_ne) * gOut;
          giy += sw_val * (ix_ne - ix) * gOut;
        }
        if (InBounds(iy_se, ix_se, in_h, in_w)) {
          T se_val = input[inp_offset_NC + iy_se * inp_sH + ix_se * inp_sW];
          gix += se_val * (iy - iy_nw) * gOut;
          giy += se_val * (ix - ix_nw) * gOut;
        }
      }

      if (grad_grid != nullptr) {
        T* gGrid_ptr_NHW = grad_grid + index * grid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      }
    } else if (mode == Mode::nearest) {
      IndexT ix_nearest = static_cast<IndexT>(std::nearbyint(ix));
      IndexT iy_nearest = static_cast<IndexT>(std::nearbyint(iy));

      IndexT gOut_offset = n * gOut_sN + h * gOut_sH + w * gOut_sW;
      T* gInp_ptr_NC = grad_input + n * inp_sN;
      for (IndexT c = 0; c < out_c;
           ++c, gInp_ptr_NC += inp_sC, gOut_offset += gOut_sC) {
        AtomicAdd(gInp_ptr_NC,
                  iy_nearest,
                  ix_nearest,
                  inp_sH,
                  inp_sW,
                  in_h,
                  in_w,
                  grad_output[gOut_offset]);
      }

      if (grad_grid != nullptr) {
        T* gGrid_ptr_NHW = grad_grid + index * grid_sW;
        gGrid_ptr_NHW[0] = static_cast<T>(0);
        gGrid_ptr_NHW[1] = static_cast<T>(0);
      }
    }
  }
}

template <typename T, typename IndexT>
__global__ void GridSampler3DCudaBackwardKernel(const IndexT nthreads,
                                                const T* grad_output,
                                                const T* input,
                                                const T* grid,
                                                IndexT out_c,
                                                IndexT out_d,
                                                IndexT out_h,
                                                IndexT out_w,
                                                IndexT in_d,
                                                IndexT in_h,
                                                IndexT in_w,
                                                T* grad_input,
                                                T* grad_grid,
                                                const Mode mode,
                                                const PaddingMode padding_mode,
                                                bool align_corners) {
  IndexT inp_sW = 1;
  IndexT inp_sH = in_w;
  IndexT inp_sD = in_h * in_w;
  IndexT inp_sC = in_d * inp_sD;
  IndexT inp_sN = out_c * inp_sC;

  IndexT grid_sCoor = 1;
  IndexT grid_sW = 3;
  IndexT grid_sH = out_w * grid_sW;
  IndexT grid_sD = out_h * grid_sH;
  IndexT grid_sN = out_d * grid_sD;

  IndexT gOut_sW = 1;
  IndexT gOut_sH = out_w;
  IndexT gOut_sD = out_h * out_w;
  IndexT gOut_sC = out_d * gOut_sD;
  IndexT gOut_sN = out_c * gOut_sC;

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, IndexT) {
    const IndexT w = index % out_w;
    const IndexT h = (index / out_w) % out_h;
    const IndexT d = (index / (out_h * out_w)) % out_d;
    const IndexT n = index / (out_d * out_h * out_w);
    const auto grid_offset =
        n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z coordinates from grid
    T ix = grid[grid_offset];
    T iy = grid[grid_offset + grid_sCoor];
    T iz = grid[grid_offset + 2 * grid_sCoor];

    // multipliers for gradients on ix, iy, and iz
    T gix_mult, giy_mult, giz_mult;
    ix = ComputePositionsWithMask(
        ix, in_w, padding_mode, align_corners, &gix_mult);
    iy = ComputePositionsWithMask(
        iy, in_h, padding_mode, align_corners, &giy_mult);
    iz = ComputePositionsWithMask(
        iz, in_d, padding_mode, align_corners, &giz_mult);

    if (mode == Mode::bilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      IndexT ix_tnw = static_cast<IndexT>(std::floor(ix));
      IndexT iy_tnw = static_cast<IndexT>(std::floor(iy));
      IndexT iz_tnw = static_cast<IndexT>(std::floor(iz));

      IndexT ix_tne = ix_tnw + 1;
      IndexT iy_tne = iy_tnw;
      IndexT iz_tne = iz_tnw;

      IndexT ix_tsw = ix_tnw;
      IndexT iy_tsw = iy_tnw + 1;
      IndexT iz_tsw = iz_tnw;

      IndexT ix_tse = ix_tnw + 1;
      IndexT iy_tse = iy_tnw + 1;
      IndexT iz_tse = iz_tnw;

      IndexT ix_bnw = ix_tnw;
      IndexT iy_bnw = iy_tnw;
      IndexT iz_bnw = iz_tnw + 1;

      IndexT ix_bne = ix_tnw + 1;
      IndexT iy_bne = iy_tnw;
      IndexT iz_bne = iz_tnw + 1;

      IndexT ix_bsw = ix_tnw;
      IndexT iy_bsw = iy_tnw + 1;
      IndexT iz_bsw = iz_tnw + 1;

      IndexT ix_bse = ix_tnw + 1;
      IndexT iy_bse = iy_tnw + 1;
      IndexT iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      T tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      T tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      T tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      T tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      T bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      T bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      T bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      T bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      T gix = static_cast<T>(0), giy = static_cast<T>(0),
        giz = static_cast<T>(0);
      IndexT gOut_offset =
          n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
      IndexT inp_offset_NC = n * inp_sN;
      T* gInp_ptr_NC = grad_input + n * inp_sN;
      for (IndexT c = 0; c < out_c; ++c,
                  gOut_offset += gOut_sC,
                  gInp_ptr_NC += inp_sC,
                  inp_offset_NC += inp_sC) {
        T gOut = grad_output[gOut_offset];

        AtomicAdd3D(gInp_ptr_NC,
                    iz_tnw,
                    iy_tnw,
                    ix_tnw,
                    inp_sD,
                    inp_sH,
                    inp_sW,
                    in_d,
                    in_h,
                    in_w,
                    tnw * gOut);
        AtomicAdd3D(gInp_ptr_NC,
                    iz_tne,
                    iy_tne,
                    ix_tne,
                    inp_sD,
                    inp_sH,
                    inp_sW,
                    in_d,
                    in_h,
                    in_w,
                    tne * gOut);
        AtomicAdd3D(gInp_ptr_NC,
                    iz_tsw,
                    iy_tsw,
                    ix_tsw,
                    inp_sD,
                    inp_sH,
                    inp_sW,
                    in_d,
                    in_h,
                    in_w,
                    tsw * gOut);
        AtomicAdd3D(gInp_ptr_NC,
                    iz_tse,
                    iy_tse,
                    ix_tse,
                    inp_sD,
                    inp_sH,
                    inp_sW,
                    in_d,
                    in_h,
                    in_w,
                    tse * gOut);
        AtomicAdd3D(gInp_ptr_NC,
                    iz_bnw,
                    iy_bnw,
                    ix_bnw,
                    inp_sD,
                    inp_sH,
                    inp_sW,
                    in_d,
                    in_h,
                    in_w,
                    bnw * gOut);
        AtomicAdd3D(gInp_ptr_NC,
                    iz_bne,
                    iy_bne,
                    ix_bne,
                    inp_sD,
                    inp_sH,
                    inp_sW,
                    in_d,
                    in_h,
                    in_w,
                    bne * gOut);
        AtomicAdd3D(gInp_ptr_NC,
                    iz_bsw,
                    iy_bsw,
                    ix_bsw,
                    inp_sD,
                    inp_sH,
                    inp_sW,
                    in_d,
                    in_h,
                    in_w,
                    bsw * gOut);
        AtomicAdd3D(gInp_ptr_NC,
                    iz_bse,
                    iy_bse,
                    ix_bse,
                    inp_sD,
                    inp_sH,
                    inp_sW,
                    in_d,
                    in_h,
                    in_w,
                    bse * gOut);

        // calculate grad_grid
        if (InBounds3D(iz_tnw, iy_tnw, ix_tnw, in_d, in_h, in_w)) {
          T tnw_val = input[inp_offset_NC + iz_tnw * inp_sD + iy_tnw * inp_sH +
                            ix_tnw * inp_sW];
          gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut;
          giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut;
          giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut;
        }
        if (InBounds3D(iz_tne, iy_tne, ix_tne, in_d, in_h, in_w)) {
          T tne_val = input[inp_offset_NC + iz_tne * inp_sD + iy_tne * inp_sH +
                            ix_tne * inp_sW];
          gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
          giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
          giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
        }
        if (InBounds3D(iz_tsw, iy_tsw, ix_tsw, in_d, in_h, in_w)) {
          T tsw_val = input[inp_offset_NC + iz_tsw * inp_sD + iy_tsw * inp_sH +
                            ix_tsw * inp_sW];
          gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut;
          giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut;
          giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut;
        }
        if (InBounds3D(iz_tse, iy_tse, ix_tse, in_d, in_h, in_w)) {
          T tse_val = input[inp_offset_NC + iz_tse * inp_sD + iy_tse * inp_sH +
                            ix_tse * inp_sW];
          gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
          giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
          giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
        }
        if (InBounds3D(iz_bnw, iy_bnw, ix_bnw, in_d, in_h, in_w)) {
          T bnw_val = input[inp_offset_NC + iz_bnw * inp_sD + iy_bnw * inp_sH +
                            ix_bnw * inp_sW];
          gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut;
          giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut;
          giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut;
        }
        if (InBounds3D(iz_bne, iy_bne, ix_bne, in_d, in_h, in_w)) {
          T bne_val = input[inp_offset_NC + iz_bne * inp_sD + iy_bne * inp_sH +
                            ix_bne * inp_sW];
          gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
          giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
          giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
        }
        if (InBounds3D(iz_bsw, iy_bsw, ix_bsw, in_d, in_h, in_w)) {
          T bsw_val = input[inp_offset_NC + iz_bsw * inp_sD + iy_bsw * inp_sH +
                            ix_bsw * inp_sW];
          gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut;
          giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut;
          giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut;
        }
        if (InBounds3D(iz_bse, iy_bse, ix_bse, in_d, in_h, in_w)) {
          T bse_val = input[inp_offset_NC + iz_bse * inp_sD + iy_bse * inp_sH +
                            ix_bse * inp_sW];
          gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
          giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
          giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
        }
      }
      if (grad_grid != nullptr) {
        T* gGrid_ptr_NDHW = grad_grid + index * grid_sW;
        gGrid_ptr_NDHW[0] = gix_mult * gix;
        gGrid_ptr_NDHW[1] = giy_mult * giy;
        gGrid_ptr_NDHW[2] = giz_mult * giz;
      }
    } else if (mode == Mode::nearest) {
      IndexT ix_nearest = static_cast<IndexT>(std::round(ix));
      IndexT iy_nearest = static_cast<IndexT>(std::round(iy));
      IndexT iz_nearest = static_cast<IndexT>(std::round(iz));

      // assign nearest neighbor pixel value to output pixel
      IndexT gOut_offset =
          n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
      T* gInp_ptr_NC = grad_input + n * inp_sN;
      for (IndexT c = 0; c < out_c;
           ++c, gOut_offset += gOut_sC, gInp_ptr_NC += inp_sC) {
        AtomicAdd3D(gInp_ptr_NC,
                    iz_nearest,
                    iy_nearest,
                    ix_nearest,
                    inp_sD,
                    inp_sH,
                    inp_sW,
                    in_d,
                    in_h,
                    in_w,
                    grad_output[gOut_offset]);
      }
      if (grad_grid != nullptr) {
        T* gGrid_ptr_NDHW = grad_grid + index * grid_sW;
        gGrid_ptr_NDHW[0] = static_cast<T>(0);
        gGrid_ptr_NDHW[1] = static_cast<T>(0);
        gGrid_ptr_NDHW[2] = static_cast<T>(0);
      }
    }
  }
}

template <typename T, typename Context>
void GridSampleGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& grid,
                          const DenseTensor& out_grad,
                          const std::string& mode,
                          const std::string& padding_mode,
                          bool align_corners,
                          DenseTensor* x_grad,
                          DenseTensor* grid_grad) {
  if (out_grad.numel() == 0) {
    if (x_grad) {
      phi::Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(x_grad->dims())), 0, x_grad);
    }
    if (grid_grad) {
      phi::Full<T, Context>(dev_ctx,
                            phi::IntArray(common::vectorize(grid_grad->dims())),
                            0,
                            grid_grad);
    }
    return;
  }

  PaddingMode enum_padding_mode;
  Mode enum_mode;
  if (padding_mode == "border") {
    enum_padding_mode = PaddingMode::border;
  } else if (padding_mode == "reflection") {
    enum_padding_mode = PaddingMode::reflect;
  } else {
    enum_padding_mode = PaddingMode::zeros;
  }

  if (mode == "nearest") {
    enum_mode = Mode::nearest;
  } else {
    enum_mode = Mode::bilinear;
  }

  bool use_int32_index = x.numel() <= std::numeric_limits<int>::max() &&
                         grid.numel() <= std::numeric_limits<int>::max() &&
                         out_grad.numel() <= std::numeric_limits<int>::max();

  if (x.dims().size() == 4) {
    const int64_t n = grid.dims()[0];
    const int64_t out_h = grid.dims()[1];
    const int64_t out_w = grid.dims()[2];
    const int64_t c = x.dims()[1];
    const int64_t in_h = x.dims()[2];
    const int64_t in_w = x.dims()[3];

    dev_ctx.template Alloc<T>(x_grad);
    phi::funcs::SetConstant<Context, T>()(dev_ctx, x_grad, static_cast<T>(0));

    T* grid_grad_data = nullptr;
    if (grid_grad != nullptr) {
      grid_grad_data = dev_ctx.template Alloc<T>(grid_grad);
    }

    int64_t count = n * out_h * out_w;
    auto cu_stream = dev_ctx.stream();
    backends::gpu::GpuLaunchConfig config =
        backends::gpu::GetGpuLaunchConfig1D(dev_ctx, count);

#define LAUNCH_KERNEL(INDEX_TYPE)                                         \
  GridSamplerCudaBackwardKernel<T, INDEX_TYPE>                            \
      <<<config.block_per_grid, config.thread_per_block, 0, cu_stream>>>( \
          count,                                                          \
          out_grad.data<T>(),                                             \
          x.data<T>(),                                                    \
          grid.data<T>(),                                                 \
          n,                                                              \
          c,                                                              \
          out_h,                                                          \
          out_w,                                                          \
          in_h,                                                           \
          in_w,                                                           \
          x_grad->data<T>(),                                              \
          grid_grad_data,                                                 \
          enum_mode,                                                      \
          enum_padding_mode,                                              \
          align_corners);
    if (use_int32_index) {
      LAUNCH_KERNEL(int32_t)
    } else {
      LAUNCH_KERNEL(int64_t)
    }
#undef LAUNCH_KERNEL
  } else {
    const int64_t out_d = grid.dims()[1];
    const int64_t out_h = grid.dims()[2];
    const int64_t out_w = grid.dims()[3];
    const int64_t n = x.dims()[0];
    const int64_t c = x.dims()[1];
    const int64_t in_d = x.dims()[2];
    const int64_t in_h = x.dims()[3];
    const int64_t in_w = x.dims()[4];

    dev_ctx.template Alloc<T>(x_grad);
    phi::funcs::SetConstant<Context, T>()(dev_ctx, x_grad, static_cast<T>(0));

    T* grid_grad_data = nullptr;
    if (grid_grad != nullptr) {
      grid_grad_data = dev_ctx.template Alloc<T>(grid_grad);
    }

    int64_t count = static_cast<int64_t>(n * out_d * out_h * out_w);
    auto cu_stream = dev_ctx.stream();
    backends::gpu::GpuLaunchConfig config =
        backends::gpu::GetGpuLaunchConfig1D(dev_ctx, count);

#define LAUNCH_KERNEL(INDEX_TYPE)                                         \
  GridSampler3DCudaBackwardKernel<T, INDEX_TYPE>                          \
      <<<config.block_per_grid, config.thread_per_block, 0, cu_stream>>>( \
          count,                                                          \
          out_grad.data<T>(),                                             \
          x.data<T>(),                                                    \
          grid.data<T>(),                                                 \
          c,                                                              \
          out_d,                                                          \
          out_h,                                                          \
          out_w,                                                          \
          in_d,                                                           \
          in_h,                                                           \
          in_w,                                                           \
          x_grad->data<T>(),                                              \
          grid_grad_data,                                                 \
          enum_mode,                                                      \
          enum_padding_mode,                                              \
          align_corners);
    if (use_int32_index) {
      LAUNCH_KERNEL(int32_t)
    } else {
      LAUNCH_KERNEL(int64_t)
    }
#undef LAUNCH_KERNEL
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(grid_sample_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GridSampleGradKernel,
                   float,
                   double) {}
