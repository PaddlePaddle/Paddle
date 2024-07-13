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

#include "paddle/phi/backends/xpu/enforce_xpu.h"

#include "gtest/gtest.h"

template <typename T>
bool CheckXPUStatusSuccess(T value, const std::string& msg = "success") {
  PADDLE_ENFORCE_XPU_SUCCESS(value);
  return true;
}

template <typename T>
bool CheckXPUStatusFailure(T value, const std::string& msg) {
  try {
    PADDLE_ENFORCE_XPU_SUCCESS(value);
    return false;
  } catch (common::enforce::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    std::cout << ex_msg << std::endl;
    return ex_msg.find(msg) != std::string::npos;
  }
}

template <typename T>
bool CheckXDNNStatusSuccess(T value, const std::string& msg = "success") {
  PADDLE_ENFORCE_XDNN_SUCCESS(value, "XDNN Error ");
  return true;
}

template <typename T>
bool CheckXDNNStatusFailure(T value, const std::string& msg) {
  try {
    PADDLE_ENFORCE_XDNN_SUCCESS(value, "XDNN Error ");
    return false;
  } catch (common::enforce::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    std::cout << ex_msg << std::endl;
    return ex_msg.find(msg) != std::string::npos;
  }
}

TEST(enforce, xpu_status) {
  EXPECT_TRUE(CheckXPUStatusSuccess(static_cast<int>(XPU_SUCCESS)));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_INVALID_DEVICE),
                                    "Invalid XPU device"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_UNINIT),
                                    "XPU runtime not properly inited"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_NOMEM),
                                    "Device memory not enough"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_NOCPUMEM),
                                    "CPU memory not enough"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_INVALID_PARAM),
                                    "Invalid parameter"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_NOXPUFUNC),
                                    "Cannot get XPU Func"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_LDSO),
                                    "Error loading dynamic library"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_LDSYM),
                                    "Error loading func from dynamic library"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_SIMULATOR),
                                    "Error from XPU Simulator"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_NOSUPPORT),
                                    "Operation not supported"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_ABNORMAL),
                                    "Device abnormal due to previous error"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_KEXCEPTION),
                                    "Exception in kernel execution"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_TIMEOUT),
                                    "Kernel execution timed out"));
  EXPECT_TRUE(
      CheckXPUStatusFailure(static_cast<int>(XPUERR_BUSY), "Resource busy"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_USEAFCLOSE),
                                    "Use a stream after closed"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_UCECC),
                                    "Uncorrectable ECC"));
  EXPECT_TRUE(
      CheckXPUStatusFailure(static_cast<int>(XPUERR_OVERHEAT), "Overheat"));
  EXPECT_TRUE(
      CheckXPUStatusFailure(static_cast<int>(XPUERR_UNEXPECT),
                            "Execution error, reach unexpected control flow"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_DEVRESET),
                                    "Device is being reset, try again later"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_HWEXCEPTION),
                                    "Hardware module exception"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_HBM_INIT),
                                    "Error init HBM"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_DEVINIT),
                                    "Error init device"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_PEERRESET),
                                    "Device is being reset, try again later"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_MAXDEV),
                                    "Device count exceed limit"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_NOIOC),
                                    "Unknown IOCTL command"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_DMATIMEOUT),
                                    "DMA timed out, a reboot maybe needed"));
  EXPECT_TRUE(CheckXPUStatusFailure(
      static_cast<int>(XPUERR_DMAABORT),
      "DMA aborted due to error, possibly wrong address or hardware state"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_MCUUNINIT),
                                    "Firmware not initialized"));
  EXPECT_TRUE(
      CheckXPUStatusFailure(static_cast<int>(XPUERR_OLDFW),
                            "Firmware version too old (<15), please update."));
  EXPECT_TRUE(
      CheckXPUStatusFailure(static_cast<int>(XPUERR_PCIE), "Error in PCIE"));
  EXPECT_TRUE(
      CheckXPUStatusFailure(static_cast<int>(XPUERR_FAULT),
                            "Error copy between kernel and user space"));
  EXPECT_TRUE(CheckXPUStatusFailure(static_cast<int>(XPUERR_INTERRUPTED),
                                    "Execution interrupted by user"));
}

#ifdef PADDLE_WITH_XPU_BKCL
TEST(enforce, bkcl_status) {
  EXPECT_TRUE(CheckXPUStatusSuccess(BKCL_SUCCESS));
  EXPECT_TRUE(
      CheckXPUStatusFailure(BKCL_INVALID_ARGUMENT, "BKCL_INVALID_ARGUMENT"));
  EXPECT_TRUE(CheckXPUStatusFailure(BKCL_RUNTIME_ERROR, "BKCL_RUNTIME_ERROR"));
  EXPECT_TRUE(CheckXPUStatusFailure(BKCL_SYSTEM_ERROR, "BKCL_SYSTEM_ERROR"));
  EXPECT_TRUE(
      CheckXPUStatusFailure(BKCL_INTERNAL_ERROR, "BKCL_INTERNAL_ERROR"));
}
#endif

TEST(enforce, xdnn_status) {
  EXPECT_TRUE(CheckXDNNStatusSuccess(xpu::Error_t::SUCCESS));
  EXPECT_TRUE(CheckXDNNStatusFailure(xpu::Error_t::INVALID_PARAM,
                                     "XDNN_INVALID_PARAM"));
  EXPECT_TRUE(CheckXDNNStatusFailure(xpu::Error_t::RUNTIME_ERROR,
                                     "XDNN_RUNTIME_ERROR"));
  EXPECT_TRUE(CheckXDNNStatusFailure(xpu::Error_t::NO_ENOUGH_WORKSPACE,
                                     "XDNN_NO_ENOUGH_WORKSPACE"));
  EXPECT_TRUE(CheckXDNNStatusFailure(xpu::Error_t::NOT_IMPLEMENT,
                                     "XDNN_NOT_IMPLEMENT"));
}
