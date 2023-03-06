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

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"


DECLARE_bool(binary_api_dump);
DECLARE_string(api_dump_list);

std::atomic<size_t> api_call_count(0);

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CSplitOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(platform::errors::Unavailable(
        "Do not support c_split for cpu kernel now."));
  }
};

bool APINeedsToBeDumped(const std::string& api_name) {
  static std::vector<std::string> api_dump_list =
      paddle::string::Split(FLAGS_api_dump_list, ',');
  if (api_dump_list.size()) {
    return std::find(api_dump_list.cbegin(), api_dump_list.cend(), api_name) !=
           api_dump_list.cend();
  }
  return true;
}

void DumpCpuTensor(const std::string& output_dir,
                   const std::string& kernel_name,
                   const std::string& tensor_name,
                   const phi::DenseTensor& cpu_tensor) {
  if (!cpu_tensor.initialized()) {
    return;
  }

  static std::unordered_map<std::string, size_t> tensor_count;

  std::stringstream ss;
  ss << output_dir << "/" << kernel_name << "/" << tensor_name;
  std::string filename = ss.str();

  ss.str("");
  ss << "mkdir -p " << filename;
  system(ss.str().c_str());

  ss.str("");
  ss << filename << "/" << tensor_count[filename]++;
  filename = ss.str();

  ss.str("");
  ss << "dims: " << cpu_tensor.dims() << std::endl;
  ss << "dtype: " << cpu_tensor.dtype() << std::endl;
  ss << "layout: " << cpu_tensor.layout() << std::endl;
  ss << "data: \n";
  if (!FLAGS_binary_api_dump) {
    auto dtype = cpu_tensor.dtype();
    auto* data = cpu_tensor.data();
    for (auto i = 0; i < cpu_tensor.numel(); ++i) {
      if (dtype == phi::DataType::FLOAT32) {
        ss << *(reinterpret_cast<const float*>(data) + i);
      } else if (dtype == phi::DataType::FLOAT16) {
        ss << static_cast<float>(
            *(reinterpret_cast<const phi::dtype::float16*>(data) + i));
      } else if (dtype == phi::DataType::FLOAT16) {
        ss << *(reinterpret_cast<const double*>(data) + i);
      } else if (dtype == phi::DataType::INT16) {
        ss << *(reinterpret_cast<const int16_t*>(data) + i);
      } else if (dtype == phi::DataType::INT32) {
        ss << *(reinterpret_cast<const int32_t*>(data) + i);
      } else if (dtype == phi::DataType::INT64) {
        ss << *(reinterpret_cast<const int64_t*>(data) + i);
      } else if (dtype == phi::DataType::BOOL) {
        ss << *(reinterpret_cast<const bool*>(data) + i);
      } else if (dtype == phi::DataType::UINT8) {
        ss << *(reinterpret_cast<const uint8_t*>(data) + i);
      } else {
        PADDLE_THROW(phi::errors::Unavailable(
            "Unsupport to dump tensors of type %d", dtype));
      }
    }
    ss << "\n";
  }

#ifdef _LINUX
  auto fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC);
  if (fd > 0) {
    if (!FLAGS_binary_api_dump) {
      write(fd, ss.str().c_str(), ss.str().size());
    } else {
      write(fd, ss.str().c_str(), ss.str().size());
      write(fd,
            cpu_tensor.data(),
            cpu_tensor.numel() *
                paddle::experimental::SizeOf(cpu_tensor.dtype()));
    }
    close(fd);
  } else {
    LOG(ERROR) << "Failed to open " << filename;
  }
#else
  std::ofstream ofs(filename.c_str());
  if (ofs) {
    if (!FLAGS_binary_api_dump) {
      ofs << ss.str();
    } else {
      ofs << ss.str();
      ofs.write(reinterpret_cast<const char*>(cpu_tensor.data()),
                cpu_tensor.numel() *
                    paddle::experimental::SizeOf(cpu_tensor.dtype()));
    }
  } else {
    LOG(ERROR) << "Failed to open " << filename;
  }
#endif
}

void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const phi::SelectedRows* tensor) {
  if (tensor) {
    DumpTensor(output_dir, kernel_name, tensor_name, *tensor);
  }
}

void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const std::shared_ptr<phi::SelectedRows>& tensor) {
  if (tensor.get()) {
    DumpTensor(output_dir, kernel_name, tensor_name, *tensor);
  }
}

void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const phi::DenseTensor* tensor) {
  if (tensor) {
    DumpTensor(output_dir, kernel_name, tensor_name, *tensor);
  }
}

void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const std::shared_ptr<phi::DenseTensor>& tensor) {
  if (tensor.get()) {
    DumpTensor(output_dir, kernel_name, tensor_name, *tensor);
  }
}

void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const phi::DenseTensor& tensor) {
  if (!tensor.initialized()) {
    return;
  }
  if (tensor.place().GetType() == phi::AllocationType::CPU) {
    DumpCpuTensor(output_dir, kernel_name, tensor_name, tensor);
  } else {
    phi::DeviceContextPool::Instance().Get(tensor.place())->Wait();
    phi::DenseTensor cpu_tensor;
    cpu_tensor.Resize(tensor.dims());
    paddle::platform::CPUPlace cpu_place;
    paddle::platform::DeviceContextPool::Instance().Get(cpu_place)->Alloc(
        &cpu_tensor, tensor.dtype());
    paddle::memory::Copy(
        cpu_place,
        cpu_tensor.data(),
        tensor.place(),
        tensor.data(),
        tensor.numel() * paddle::experimental::SizeOf(tensor.dtype()));
    phi::DeviceContextPool::Instance().Get(tensor.place())->Wait();
    DumpCpuTensor(output_dir, kernel_name, tensor_name, cpu_tensor);
  }
}

void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const std::vector<const phi::DenseTensor*>& tensor_vec) {
  for (size_t i = 0; i < tensor_vec.size(); ++i) {
    std::stringstream ss;
    ss << tensor_name << "_" << i;
    DumpTensor(output_dir, kernel_name, ss.str(), *tensor_vec[i]);
  }
}

void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const std::vector<phi::DenseTensor*>& tensor_vec) {
  for (size_t i = 0; i < tensor_vec.size(); ++i) {
    std::stringstream ss;
    ss << tensor_name << "_" << i;
    DumpTensor(output_dir, kernel_name, ss.str(), *tensor_vec[i]);
  }
}

void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const phi::SelectedRows& tensor) {}
void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const paddle::optional<phi::DenseTensor>& tensor) {}
void DumpTensor(
    const std::string& output_dir,
    const std::string& kernel_name,
    const std::string& tensor_name,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& tensor) {}

void DumpTensor(const std::string& output_dir,
                const std::string& kernel_name,
                const std::string& tensor_name,
                const paddle::optional<phi::SelectedRows>& tensor) {}

}  // namespace experimental
}  // namespace paddle
