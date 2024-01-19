#pragma once

#include <memory>
#include <optional>
#include "paddle/pir/pass/pass.h"


namespace cinn::dialect::ir {

// This is a helper pass for preparing dynamic-shape inputs to cinn backend even in static shape GroupOp.
// Returns std::nullopt if FLAGS_cinn_convert_static_dim_to_dynamic not set or invalid.
std::optional<std::unique_ptr<::pir::Pass>> CreateConvertStaticDimToDynamicPass();

}