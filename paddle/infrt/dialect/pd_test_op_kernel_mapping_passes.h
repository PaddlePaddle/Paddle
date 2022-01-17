#pragma once
#include <mlir/Pass/Pass.h>
#include <memory>

namespace infrt {

//! Create a Pass to align Op and Kernel.
std::unique_ptr<mlir::Pass> CreateOpKernelMapingPass();

}  // namespace infrt

