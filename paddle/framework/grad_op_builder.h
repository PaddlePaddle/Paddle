#pragma once

#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {

OperatorBase* BuildGradOp(const OperatorBase* op);

}  // namespace framework
}  // namespace paddle
