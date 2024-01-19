#pragma once

#include <memory>
#include <optional>
#include "paddle/pir/pass/pass.h"


namespace cinn::dialect::ir {

std::optional<std::unique_ptr<::pir::Pass>> CreateConvertStaticDimToDynamicPass();

}