/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cstdint>
#include <fstream>
#include <numeric>

#include "glog/logging.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/phi/common/port.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"

namespace pir {

const phi::DeviceContext* GetDeviceContext(
    const phi::DenseTensor& x, const phi::Place& place = phi::Place()) {
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  const phi::DeviceContext* dev_ctx = nullptr;
  auto x_place = x.place();
  if (x_place.GetType() != phi::AllocationType::UNDEFINED) {
    dev_ctx = pool.Get(x_place);
    return dev_ctx;
  } else if (place.GetType() != phi::AllocationType::UNDEFINED) {
    dev_ctx = pool.Get(place);
    return dev_ctx;
  } else {
    phi::Place compile_place;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    compile_place = phi::GPUPlace();
#elif defined(PADDLE_WITH_XPU)
    compile_place = phi::XPUPlace();
#else
    compile_place = phi::CPUPlace();
#endif
    dev_ctx = pool.Get(compile_place);
    return dev_ctx;
  }
  return dev_ctx;
}

const phi::DenseTensor CastTensorType(const phi::DeviceContext* dev_ctx,
                                      const phi::DenseTensor& x,
                                      phi::DataType out_dtype) {
  auto place = x.place();
  if (phi::is_cpu_place(place)) {
    auto out = phi::funcs::TransDataType(
        reinterpret_cast<const phi::CPUContext&>(*dev_ctx), x, out_dtype);
    return out;
  } else if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    return phi::funcs::TransDataType(
        reinterpret_cast<const phi::GPUContext&>(*dev_ctx), x, out_dtype);
#endif
  }
  return x;
}

void SaveFunction(const phi::DenseTensor& x,
                  const std::string& name,
                  const std::string& file_path,
                  bool overwrite,
                  bool save_as_fp16) {
  PADDLE_ENFORCE_EQ(
      FileExists(file_path) && !overwrite,
      false,
      common::errors::PreconditionNotMet(
          "%s exists!, cannot save to it when overwrite is set to false.",
          file_path,
          overwrite));

  MkDirRecursively(DirName(file_path).c_str());
  VLOG(6) << "save func save path: " << file_path;
  std::ofstream fout(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    common::errors::Unavailable(
                        "Cannot open %s to save variables.", file_path));

  auto in_dtype = x.dtype();
  auto out_dtype = save_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;

  const phi::DeviceContext* dev_ctx = GetDeviceContext(x);
  if (in_dtype != out_dtype) {
    auto out = CastTensorType(dev_ctx, x, out_dtype);
    phi::SerializeToStream(fout, out, *dev_ctx);
  } else {
    phi::SerializeToStream(fout, x, *dev_ctx);
  }
  fout.close();
  VLOG(6) << "save func done ";
}

void SaveCombineFunction(const std::vector<const phi::DenseTensor*>& x,
                         const std::vector<std::string>& names,
                         const std::string& file_path,
                         bool overwrite,
                         bool save_as_fp16,
                         bool save_to_memory) {
  PADDLE_ENFORCE_EQ(
      FileExists(file_path) && !overwrite,
      false,
      common::errors::PreconditionNotMet(
          "%s exists!, cannot save to it when overwrite is set to false.",
          file_path,
          overwrite));

  MkDirRecursively(DirName(file_path).c_str());
  VLOG(6) << "save func save path: " << file_path;
  std::ofstream fout(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    common::errors::Unavailable(
                        "Cannot open %s to save variables.", file_path));
  PADDLE_ENFORCE_GT(x.size(),
                    0UL,
                    common::errors::InvalidArgument(
                        "The number of variables to be saved is %d, expect "
                        "it to be greater than 0.",
                        x.size()));
  const phi::DeviceContext* dev_ctx = GetDeviceContext(*(x[0]));
  for (size_t i = 0; i < x.size(); i++) {
    auto& tensor = *(x[i]);
    PADDLE_ENFORCE_EQ(
        tensor.IsInitialized(),
        true,
        common::errors::InvalidArgument(
            "The Tensor with Index (%d) to be saved is not initialized.", i));
    auto in_dtype = tensor.dtype();
    auto out_dtype = save_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;
    if (in_dtype != out_dtype) {
      auto out = CastTensorType(dev_ctx, tensor, out_dtype);
      phi::SerializeToStream(fout, out, *dev_ctx);
    } else {
      phi::SerializeToStream(fout, tensor, *dev_ctx);
    }
  }
  fout.close();
  VLOG(6) << "save combine done ";
}

void LoadFunction(const std::string& file_path,
                  int64_t seek,
                  const std::vector<int64_t>& shape,
                  bool load_as_fp16,
                  phi::DenseTensor* out,
                  phi::Place place) {
  std::ifstream fin(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fin),
                    true,
                    common::errors::Unavailable(
                        "Load operator fail to open file %s, please check "
                        "whether the model file is complete or damaged.",
                        file_path));
  PADDLE_ENFORCE_NOT_NULL(out,
                          common::errors::InvalidArgument(
                              "The variable to be loaded cannot be found."));
  const phi::DeviceContext* dev_ctx = GetDeviceContext(*out, place);

  if (seek != -1) {
    PADDLE_ENFORCE_GE(seek,
                      0,
                      common::errors::InvalidArgument(
                          "seek with tensor must great than or equal to 0"));
    phi::DeserializeFromStream(fin, out, *dev_ctx, seek, shape);
  } else {
    phi::DeserializeFromStream(fin, out, *dev_ctx);
  }

  auto in_dtype = out->dtype();
  auto out_dtype = load_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;
  if (in_dtype != out_dtype) {
    auto cast_in = *out;
    *out = CastTensorType(dev_ctx, cast_in, out_dtype);
  }
}

void LoadCombineFunction(const std::string& file_path,
                         const std::vector<std::string>& names,
                         std::vector<phi::DenseTensor*>* out,
                         bool load_as_fp16,
                         phi::Place place) {
  std::ifstream fin(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fin),
                    true,
                    common::errors::Unavailable(
                        "Load operator fail to open file %s, please check "
                        "whether the model file is complete or damaged.",
                        file_path));

  PADDLE_ENFORCE_GT(out->size(),
                    0UL,
                    common::errors::InvalidArgument(
                        "The number of variables to be saved is %d, expect "
                        "it to be greater than 0.",
                        out->size()));
  const phi::DeviceContext* dev_ctx = GetDeviceContext(*(out->at(0)), place);
  for (size_t i = 0; i < names.size(); i++) {
    auto tensor = out->at(i);
    phi::DeserializeFromStream(fin, tensor, *dev_ctx);

    auto in_dtype = tensor->dtype();
    auto out_dtype = load_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;
    if (in_dtype != out_dtype) {
      auto cast_in = *tensor;
      *tensor = CastTensorType(dev_ctx, cast_in, out_dtype);
    }
  }
  fin.peek();
  PADDLE_ENFORCE_EQ(fin.eof(),
                    true,
                    common::errors::Unavailable(
                        "Not allowed to load partial data via "
                        "load_combine_op, please use load_op instead."));
}

}  // namespace pir
