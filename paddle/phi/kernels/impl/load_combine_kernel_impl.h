// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/framework/convert_utils.h"
#include "paddle/phi/core/framework/data_type_transform.h"
#include "paddle/phi/core/framework/dense_tensor_serialize.h"
#include "paddle/phi/core/framework/var_type_helper.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/raw_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/vocab/string_array.h"

namespace phi {

template <typename T, typename Context>
void LoadParamsFromBuffer(const Context& dev_ctx,
                          const phi::Place& place,
                          std::istream* buffer,
                          bool load_as_fp16,
                          const std::vector<phi::DenseTensor*>& out) {
  auto out_vars = out;
  for (size_t i = 0; i < out_vars.size(); i++) {
    PADDLE_ENFORCE_NOT_NULL(
        out_vars[i],
        common::errors::InvalidArgument(
            "The variable index %d to be loaded cannot be found.", i));
    // Error checking
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(*buffer),
        true,
        common::errors::Unavailable(
            "An error occurred while loading model parameters. "
            "Please check whether the model file is complete or damaged."));

    dev_ctx.template Alloc<T>(out_vars[i]);
    phi::DenseTensor* tensor = out_vars[i];
    // Get data from fin to tensor
    phi::DeserializeFromStream(*buffer, tensor, dev_ctx);
    auto in_dtype = tensor->dtype();
    auto out_dtype = load_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;
    if (in_dtype != out_dtype) {
      // convert to float16 tensor
      auto in_kernel_type =
          phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, in_dtype);
      auto out_kernel_type =
          phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, out_dtype);
      phi::DenseTensor fp16_tensor;
      // copy LoD info to the new tensor
      fp16_tensor.set_lod(tensor->lod());
      TransDataType(in_kernel_type, out_kernel_type, *tensor, &fp16_tensor);

      // reset output tensor
      tensor->set_lod(fp16_tensor.lod());
      tensor->ShareDataWith(fp16_tensor);
    }
  }
  buffer->peek();
  PADDLE_ENFORCE_EQ(buffer->eof(),
                    true,
                    common::errors::Unavailable(
                        "Not allowed to load partial data via "
                        "load_combine_op, please use load_op instead."));
}

template <typename T, typename Context>
void LoadParamsFromBuffer(const Context& dev_ctx,
                          const phi::Place& place,
                          std::istream* buffer,
                          bool load_as_fp16,
                          const std::vector<phi::Vocab*>& out) {
  auto out_vars = out;
  for (size_t i = 0; i < out_vars.size(); i++) {
    PADDLE_ENFORCE_NOT_NULL(
        out_vars[i],
        common::errors::InvalidArgument(
            "The variable index %d to be loaded cannot be found.", i));
    // Error checking
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(*buffer),
        true,
        common::errors::Unavailable(
            "An error occurred while loading model parameters. "
            "Please check whether the model file is complete or damaged."));

    auto* tensor = out_vars[i];
    tensor->clear();
    std::unordered_map<std::string, std::int32_t> data;
    StringMapFromStream(*buffer, &data);
    for (auto it = data.begin(); it != data.end(); ++it) {
      std::string tmp;
      NFD(it->first, &tmp);
      if (tmp.empty()) {
        // VLOG(0) << "The string " << it->first
        //         << " was converted to unicode unsuccessfully! "
        //         << "Then dropped to load it.";
        continue;
      }
      std::wstring token;
      bool status = ConvertStrToWstr(tmp, &token);
      if (!status) continue;
      tensor->emplace(token, it->second);
    }
  }
  buffer->peek();
  PADDLE_ENFORCE_EQ(buffer->eof(),
                    true,
                    common::errors::Unavailable(
                        "Not allowed to load partial data via "
                        "load_combine_op, please use load_op instead."));
}

template <typename T, typename Context>
void LoadParamsFromBuffer(const Context& dev_ctx,
                          const phi::Place& place,
                          std::istream* buffer,
                          bool load_as_fp16,
                          const std::vector<phi::ExtendedTensor*>& out) {
  auto out_vars = out;
  for (size_t i = 0; i < out_vars.size(); i++) {
    PADDLE_ENFORCE_NOT_NULL(
        out_vars[i],
        common::errors::InvalidArgument(
            "The variable index %d to be loaded cannot be found.", i));
    // Error checking
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(*buffer),
        true,
        common::errors::Unavailable(
            "An error occurred while loading model parameters. "
            "Please check whether the model file is complete or damaged."));
    auto* raw_tensor = static_cast<RawTensor*>(out_vars[i]);
    if (raw_tensor->IsType<Vocab>()) {
      auto* tensor = raw_tensor->GetMutable<Vocab>();
      tensor->clear();
      std::unordered_map<std::string, std::int32_t> data;
      StringMapFromStream(*buffer, &data);
      for (auto it = data.begin(); it != data.end(); ++it) {
        std::string tmp;
        NFD(it->first, &tmp);
        if (tmp.empty()) {
          // VLOG(0) << "The string " << it->first
          //         << " was converted to unicode unsuccessfully! "
          //         << "Then dropped to load it.";
          continue;
        }
        std::wstring token;
        bool status = ConvertStrToWstr(tmp, &token);
        if (!status) continue;
        tensor->emplace(token, it->second);
      }
    } else {
      auto* tensor = raw_tensor->GetMutable<DenseTensor>();

      // Get data from fin to tensor
      DeserializeFromStream(*buffer, tensor, dev_ctx);

      auto in_dtype = tensor->dtype();
      auto out_dtype = load_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;

      if (in_dtype != out_dtype) {
        // convert to float16 tensor
        auto in_kernel_type =
            phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, in_dtype);
        auto out_kernel_type =
            phi::KernelKey(place, phi::DataLayout::ALL_LAYOUT, out_dtype);
        phi::DenseTensor fp16_tensor;
        // copy LoD info to the new tensor
        fp16_tensor.set_lod(tensor->lod());
        TransDataType(in_kernel_type, out_kernel_type, *tensor, &fp16_tensor);

        // reset output tensor
        // raw_tensor->Clear();
        tensor = raw_tensor->GetMutable<phi::DenseTensor>();
        tensor->set_lod(fp16_tensor.lod());
        tensor->ShareDataWith(fp16_tensor);
      }
    }
  }
  buffer->peek();
  PADDLE_ENFORCE_EQ(buffer->eof(),
                    true,
                    common::errors::Unavailable(
                        "Not allowed to load partial data via "
                        "load_combine_op, please use load_op instead."));
}

template <typename T, typename Context>
void LoadCombineKernel(const Context& dev_ctx,
                       const std::string& file_path,
                       bool load_as_fp16,
                       bool model_from_memory,
                       std::vector<phi::DenseTensor*> out) {
  auto place = dev_ctx.GetPlace();
  auto filename = file_path;
  auto out_var_names = out;

  if (!model_from_memory) {
    std::ifstream fin(filename, std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fin),
        true,
        common::errors::Unavailable(
            "LoadCombine operator fails to open file %s, please check "
            "whether the model file is complete or damaged.",
            filename));
    LoadParamsFromBuffer<T, Context>(dev_ctx, place, &fin, load_as_fp16, out);
  } else {
    PADDLE_ENFORCE_NE(
        filename.empty(),
        true,
        common::errors::Unavailable(
            "LoadCombine operator fails to open file %s, please check "
            "whether the model file is complete or damaged.",
            filename));
    std::stringstream fin(filename, std::ios::in | std::ios::binary);
    LoadParamsFromBuffer<T, Context>(dev_ctx, place, &fin, load_as_fp16, out);
  }
}

template <typename T, typename Context>
void LoadCombineVocabKernel(const Context& dev_ctx,
                            const std::string& file_path,
                            bool load_as_fp16,
                            bool model_from_memory,
                            std::vector<phi::Vocab*> out) {
  auto place = dev_ctx.GetPlace();
  auto filename = file_path;
  auto out_var_names = out;

  PADDLE_ENFORCE_GT(out_var_names.size(),
                    0UL,
                    common::errors::InvalidArgument(
                        "The number of variables to be loaded is %d, expect "
                        "it to be greater than 0.",
                        out_var_names.size()));
  if (!model_from_memory) {
    std::ifstream fin(filename, std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fin),
        true,
        common::errors::Unavailable(
            "LoadCombine operator fails to open file %s, please check "
            "whether the model file is complete or damaged.",
            filename));
    LoadParamsFromBuffer<T, Context>(dev_ctx, place, &fin, load_as_fp16, out);
  } else {
    PADDLE_ENFORCE_NE(
        filename.empty(),
        true,
        common::errors::Unavailable(
            "LoadCombine operator fails to open file %s, please check "
            "whether the model file is complete or damaged.",
            filename));
    std::stringstream fin(filename, std::ios::in | std::ios::binary);
    LoadParamsFromBuffer<T, Context>(dev_ctx, place, &fin, load_as_fp16, out);
  }
}

template <typename T, typename Context>
void LoadCombineExtendedKernel(const Context& dev_ctx,
                               const std::string& file_path,
                               bool load_as_fp16,
                               bool model_from_memory,
                               std::vector<phi::ExtendedTensor*> out) {
  auto place = dev_ctx.GetPlace();
  auto filename = file_path;
  auto out_var_names = out;

  PADDLE_ENFORCE_GT(out_var_names.size(),
                    0UL,
                    common::errors::InvalidArgument(
                        "The number of variables to be loaded is %d, expect "
                        "it to be greater than 0.",
                        out_var_names.size()));
  if (!model_from_memory) {
    std::ifstream fin(filename, std::ios::binary);
    PADDLE_ENFORCE_EQ(
        static_cast<bool>(fin),
        true,
        common::errors::Unavailable(
            "LoadCombine operator fails to open file %s, please check "
            "whether the model file is complete or damaged.",
            filename));
    LoadParamsFromBuffer<T, Context>(dev_ctx, place, &fin, load_as_fp16, out);
  } else {
    PADDLE_ENFORCE_NE(
        filename.empty(),
        true,
        common::errors::Unavailable(
            "LoadCombine operator fails to open file %s, please check "
            "whether the model file is complete or damaged.",
            filename));
    std::stringstream fin(filename, std::ios::in | std::ios::binary);
    LoadParamsFromBuffer<T, Context>(dev_ctx, place, &fin, load_as_fp16, out);
  }
}
}  // namespace phi
