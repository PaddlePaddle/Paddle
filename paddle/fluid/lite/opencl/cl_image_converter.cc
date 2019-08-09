/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/lite/opencl/cl_image_converter.h"
#include <glog/logging.h>
#include <vector>

namespace paddle {
namespace lite {

DDim CLImageConverterDefault::InitImageDimInfoWith(const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }
  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];
  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterDefault::NCHWToImage(float *nchw, float *image,
                                          const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }

  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];

  DDim in_image_dim = InitImageDimInfoWith(tensor_dim);

  VLOG(3) << " tensor dim: " << tensor_dim;
  VLOG(3) << " image dim: " << in_image_dim;

  size_t width = in_image_dim[0];
  size_t w_block = width / W;

  float *p = nchw;
  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < w_block * 4; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          if (c < C) {
            // size_t x = (n * width * H + h * width + (c / 4) * W + w) * 4 +
            // (c % 4);
            image[i2] = *p;
            i2 += 4;
            p++;
          } else {
            image[i2] = 0.0;
            i2 += 4;
          }
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

void CLImageConverterDefault::ImageToNCHW(float *image, float *tensor,
                                          const DDim &image_dim,
                                          const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }

  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];

  size_t width = image_dim[0];
  float *p = tensor;

  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          *p = image[i2];
          i2 += 4;
          p++;
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

DDim CLImageConverterFolder::InitImageDimInfoWith(const DDim &tensor_dim) {
  if (tensor_dim.size() <= 2) {
    size_t tdim[2] = {1, 1};
    if (tensor_dim.size() == 1) {
      tdim[1] = tensor_dim[0];
    } else {
      tdim[0] = tensor_dim[0];
      tdim[1] = tensor_dim[1];
    }
    size_t width = (tdim[1] + 3) / 4;
    size_t height = tdim[0];

    width_of_one_block_ = width;
    height_of_one_block_ = height;
    c_block_ = 1;

    return DDim(
        std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                       static_cast<DDim::value_type>(height)}));

  } else {
    size_t new_dims[] = {1, 1, 1, 1};
    for (size_t j = 0; j < tensor_dim.size(); ++j) {
      new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
    }
    size_t N, C, H, W;
    N = new_dims[0];
    C = new_dims[1];
    H = new_dims[2];
    W = new_dims[3];
    size_t width = W * ((C + 3) / 4);
    size_t height = H * N;

    width_of_one_block_ = W;
    height_of_one_block_ = H;
    c_block_ = width / W;

    return DDim(
        std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                       static_cast<DDim::value_type>(height)}));
  }
}

void CLImageConverterFolder::NCHWToImage(float *tensor, float *image,
                                         const DDim &tensor_dim) {
  CHECK(tensor_dim.size() <= 4 && tensor_dim.size() > 0)
      << " Tensor dim is not support!";

  if (tensor_dim.size() > 2) {
    CLImageConverterDefault default_converter;
    default_converter.NCHWToImage(tensor, image, tensor_dim);

  } else {
    size_t tdim[2] = {1, 1};
    if (tensor_dim.size() == 1) {
      tdim[1] = tensor_dim[0];
    } else {
      tdim[0] = tensor_dim[0];
      tdim[1] = tensor_dim[1];
    }

    DDim image_dim = InitImageDimInfoWith(tensor_dim);
    size_t width = image_dim[0];

    for (size_t h = 0; h < tdim[0]; h++) {
      for (size_t w = 0; w < tdim[1]; w++) {
        image[(h * width + w / 4) * 4 + (w % 4)] = tensor[h * tdim[1] + w];
      }
    }
  }
}

void CLImageConverterFolder::ImageToNCHW(float *image, float *tensor,
                                         const DDim &image_dim,
                                         const DDim &tensor_dim) {
  if (tensor_dim.size() > 2) {
    CLImageConverterDefault default_converter;
    default_converter.ImageToNCHW(image, tensor, image_dim, tensor_dim);

  } else {
    size_t width = image_dim[0];
    size_t H = 1, W = 1;

    if (tensor_dim.size() == 2) {
      H = tensor_dim[0];
      W = tensor_dim[1];
    } else if (tensor_dim.size() == 1) {
      W = tensor_dim[0];
    }

    float *p = tensor;

    for (size_t h = 0; h < H; h++) {
      for (size_t w = 0; w < W; w++) {
        p[h * W + w] = image[(h * width + w / 4) * 4 + (w % 4)];
      }
    }
  }
}

DDim CLImageConverterNWBlock::InitImageDimInfoWith(const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];
  size_t width = W * ((N + 3) / 4);
  size_t height = C * H;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterNWBlock::NCHWToImage(float *tensor, float *image,
                                          const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  auto image_dim = InitImageDimInfoWith(tensor_dim);
  float *p = tensor;
  size_t N = tensor_dim[0];
  size_t C = tensor_dim[1];
  size_t H = tensor_dim[2];
  size_t W = tensor_dim[3];
  size_t width = image_dim[0];
  size_t height = image_dim[1];
  size_t block = image_dim[0] / tensor_dim[3];

  for (size_t n = 0; n < block * 4; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          size_t index = 4 * c * (width * H) + 4 * h * width + 4 * W * (n / 4) +
                         w * 4 + n % 4;
          if (n < N) {
            image[index] = *p;
            p++;
          } else {
            image[index] = 0.0;
          }
          if (index >= (width * height * 4)) {
            LOG(INFO) << " index out of range ";
          }
        }
      }
    }
  }
  VLOG(3) << " init done";
}

void CLImageConverterNWBlock::ImageToNCHW(float *image, float *tensor,
                                          const DDim &image_dim,
                                          const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  float *p = tensor;
  size_t N = tensor_dim[0];
  size_t C = tensor_dim[1];
  size_t H = tensor_dim[2];
  size_t W = tensor_dim[3];
  size_t width = image_dim[0];
  size_t height = image_dim[1];

  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          size_t index = 4 * c * (width * H) + 4 * h * width + 4 * W * (n / 4) +
                         w * 4 + n % 4;
          *p = image[index];
          p++;
          if (index >= (width * height * 4)) {
            LOG(INFO) << " index out of range ";
          }
        }
      }
    }
  }
  VLOG(3) << " init done";
}

DDim CLImageConverterDWBlock::InitImageDimInfoWith(const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C, H, W;
  N = tensor_dim[0];
  C = tensor_dim[1];
  H = tensor_dim[2];
  W = tensor_dim[3];
  size_t width = W * ((N + 3) / 4);
  size_t height = C * H;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterDWBlock::NCHWToImage(float *tensor, float *image,
                                          const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }

  size_t N, C, H, W;
  N = new_dims[1];
  C = new_dims[0];
  H = new_dims[2];
  W = new_dims[3];

  DDim in_image_dim = InitImageDimInfoWith(tensor_dim);

  VLOG(3) << " tensor dim: " << tensor_dim;
  VLOG(3) << " image dim: " << in_image_dim;

  size_t width = in_image_dim[0];
  size_t w_block = width / W;

  float *p = tensor;
  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < w_block * 4; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          if (c < C) {
            // size_t x = (n * width * H + h * width + (c / 4) * W + w) * 4 +
            // (c % 4);
            image[i2] = *p;
            i2 += 4;
            p++;
          } else {
            image[i2] = 0.0;
            i2 += 4;
          }
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

void CLImageConverterDWBlock::ImageToNCHW(float *image, float *tensor,
                                          const DDim &image_dim,
                                          const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  float *p = tensor;
  size_t N = tensor_dim[1];
  size_t C = tensor_dim[0];
  size_t H = tensor_dim[2];
  size_t W = tensor_dim[3];
  size_t width = image_dim[0];

  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < C; c++) {
      size_t i1 = i0 + (c / 4) * W;
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;
        for (size_t w = 0; w < W; w++) {
          *p = image[i2];
          i2 += 4;
          p++;
        }
        i1 += width;
      }
    }
    i0 += width * H;
  }
}

DDim CLImageConverterNormal::InitImageDimInfoWith(const DDim &tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }
  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];
  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;

  width_of_one_block_ = W;
  height_of_one_block_ = H;
  c_block_ = width / W;

  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterNormal::NCHWToImage(float *tensor, float *image,
                                         const DDim &tensor_dim) {
  CHECK(tensor_dim.size() <= 4 && tensor_dim.size() > 0)
      << " Tensor dim is not support!";

  CLImageConverterDefault default_converter;
  default_converter.NCHWToImage(tensor, image, tensor_dim);
}

void CLImageConverterNormal::ImageToNCHW(float *image, float *tensor,
                                         const DDim &image_dim,
                                         const DDim &tensor_dim) {
  CLImageConverterDefault default_converter;
  default_converter.ImageToNCHW(image, tensor, image_dim, tensor_dim);
}

DDim CLImageConverterWinoTransWeight::InitImageDimInfoWith(
    const DDim &tensor_dim) {
  CHECK(tensor_dim.size() == 4) << " Tensor dim is not 4.";
  size_t N, C;
  N = tensor_dim[0];
  C = tensor_dim[1];
  size_t width = (C + 3) / 4;
  size_t height = N * 16;  // N * (wino_blk_size + 2) * (wino_blk_size + 2)
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

void CLImageConverterWinoTransWeight::NCHWToImage(float *tensor, float *image,
                                                  const DDim &tensor_dim) {}

void CLImageConverterWinoTransWeight::ImageToNCHW(float *image, float *tensor,
                                                  const DDim &image_dim,
                                                  const DDim &tensor_dim) {}

}  // namespace lite
}  // namespace paddle
