// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/dialect/shape/utils/dim_expr.h"
#include "paddle/pir/core/utils.h"

namespace symbol {

DimExpr DimExpr::operator+(const DimExpr& other) const {
  if (this->isa<std::int64_t>() && other.isa<std::int64_t>()) {
    return this->dyn_cast<std::int64_t>() + other.dyn_cast<std::int64_t>();
  }
  return Add<DimExpr>{List<DimExpr>{*this, other}};
}

DimExpr DimExpr::operator-(const DimExpr& other) const {
  if (this->isa<std::int64_t>() && other.isa<std::int64_t>()) {
    return this->dyn_cast<std::int64_t>() - other.dyn_cast<std::int64_t>();
  }
  const DimExpr& neg = Negative<DimExpr>(other);
  return Add<DimExpr>{List<DimExpr>{*this, neg}};
}

DimExpr DimExpr::operator*(const DimExpr& other) const {
  if (this->isa<std::int64_t>() && other.isa<std::int64_t>()) {
    return this->dyn_cast<std::int64_t>() * other.dyn_cast<std::int64_t>();
  }
  return Mul<DimExpr>{List<DimExpr>{*this, other}};
}

DimExpr DimExpr::operator/(const DimExpr& other) const {
  if (this->isa<std::int64_t>() && other.isa<std::int64_t>()) {
    std::int64_t num = this->dyn_cast<std::int64_t>();
    std::int64_t dem = other.dyn_cast<std::int64_t>();
    if (num % dem == 0) {
      return num / dem;
    }
  }
  const DimExpr& reciprocal = Reciprocal<DimExpr>(other);
  return Mul<DimExpr>{List<DimExpr>{*this, reciprocal}};
}

namespace {

bool DimExprEqual(std::int64_t lhs, std::int64_t rhs) { return lhs == rhs; }

bool DimExprEqual(const std::string& lhs, const std::string& rhs) {
  return lhs == rhs;
}

bool DimExprEqual(const Negative<DimExpr>& lhs, const Negative<DimExpr>& rhs) {
  return lhs->data == rhs->data;
}

bool DimExprEqual(const Reciprocal<DimExpr>& lhs,
                  const Reciprocal<DimExpr>& rhs) {
  return lhs->data == rhs->data;
}

template <template <typename> class Op>
bool DimExprEqual(const Op<DimExpr>& lhs, const Op<DimExpr>& rhs) {
  if (lhs.operands->size() != rhs.operands->size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.operands->size(); ++i) {
    if (lhs.operands->at(i) != rhs.operands->at(i)) {
      return false;
    }
  }
  return true;
}

bool DimExprEqual(const Add<DimExpr>& lhs, const Add<DimExpr>& rhs) {
  return DimExprEqual<Add>(lhs, rhs);
}

bool DimExprEqual(const Mul<DimExpr>& lhs, const Mul<DimExpr>& rhs) {
  return DimExprEqual<Mul>(lhs, rhs);
}

bool DimExprEqual(const Max<DimExpr>& lhs, const Max<DimExpr>& rhs) {
  return DimExprEqual<Max>(lhs, rhs);
}

bool DimExprEqual(const Min<DimExpr>& lhs, const Min<DimExpr>& rhs) {
  return DimExprEqual<Min>(lhs, rhs);
}

bool DimExprEqual(const Broadcast<DimExpr>& lhs,
                  const Broadcast<DimExpr>& rhs) {
  return DimExprEqual<Broadcast>(lhs, rhs);
}

}  // namespace

bool DimExpr::operator==(const DimExpr& other) const {
  if (this == &other) {
    return true;
  }
  return std::visit(
      [](const auto& lhs, const auto& rhs) {
        if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>,
                                     std::decay_t<decltype(rhs)>>) {
          return DimExprEqual(lhs, rhs);
        } else {
          return false;
        }
      },
      this->variant(),
      other.variant());
}

bool DimExpr::operator!=(const DimExpr& other) const {
  return !(*this == other);
}

namespace {
template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

std::string ListDimExprToString(const List<DimExpr>& dim_exprs,
                                const std::string& delim = ", ") {
  std::string ret;
  for (std::size_t i = 0; i < dim_exprs->size(); ++i) {
    if (i > 0) {
      ret += delim;
    }
    ret += ToString(dim_exprs->at(i));
  }
  return ret;
}
}  // namespace

std::string ToString(const DimExpr& dim_expr) {
  auto lambdas = Overloaded{
      [](std::int64_t dim_expr) { return std::to_string(dim_expr); },
      [](const std::string& dim_expr) { return dim_expr; },
      [](const Negative<DimExpr>& dim_expr) {
        return "-" + ToString(dim_expr->data);
      },
      [](const Reciprocal<DimExpr>& dim_expr) {
        return "1 / (" + ToString(dim_expr->data) + ")";
      },
      [](const Add<DimExpr>& dim_expr) {
        return "Add(" + ListDimExprToString(dim_expr.operands, ", ") + ")";
      },
      [](const Mul<DimExpr>& dim_expr) {
        return "Mul(" + ListDimExprToString(dim_expr.operands, ", ") + ")";
      },
      [](const Max<DimExpr>& dim_expr) {
        return "Max(" + ListDimExprToString(dim_expr.operands, ", ") + ")";
      },
      [](const Min<DimExpr>& dim_expr) {
        return "Min(" + ListDimExprToString(dim_expr.operands, ", ") + ")";
      },
      [](const Broadcast<DimExpr>& dim_expr) {
        return "Broadcast(" + ListDimExprToString(dim_expr.operands, ", ") +
               ")";
      }};
  return std::visit(lambdas, dim_expr.variant());
}

std::ostream& operator<<(std::ostream& stream, const DimExpr& dim_expr) {
  stream << ToString(dim_expr);
  return stream;
}

std::ostream& operator<<(std::ostream& stream,
                         const std::vector<DimExpr>& dim_exprs) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < dim_exprs.size(); ++i) {
    ss << ToString(dim_exprs[i]);
    if (i < dim_exprs.size() - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return stream << ss.str();
}

std::ostream& operator<<(std::ostream& stream,
                         const ShapeOrDataDimExprs& shape_or_data) {
  std::string result;
  auto lambdas = Overloaded{
      [&result](const TensorShapeOrDataDimExprs& tensor_shape_data) {
        result += "shape[";
        for (size_t i = 0; i < tensor_shape_data.shape().size(); ++i) {
          result += ToString(tensor_shape_data.shape()[i]);
          if (i < tensor_shape_data.shape().size() - 1) {
            result += ", ";
          }
        }
        result += "]";
        if (tensor_shape_data.data()) {
          result += ", data[";
          for (size_t i = 0; i < tensor_shape_data.data()->size(); ++i) {
            result += ToString(tensor_shape_data.data()->at(i));
            if (i < tensor_shape_data.data()->size() - 1) {
              result += ", ";
            }
          }
          result += "]";
        } else {
          result += ", data[NULL]";
        }
      },
      [&result](const TensorListShapeOrDataDimExprs& tensor_list_shape_data) {
        for (size_t i = 0; i < tensor_list_shape_data.size(); ++i) {
          result += "shape[";
          for (size_t i = 0; i < tensor_list_shape_data[i].shape().size();
               ++i) {
            result += ToString(tensor_list_shape_data[i].shape()[i]);
            if (i < tensor_list_shape_data[i].shape().size() - 1) {
              result += ", ";
            }
          }
          result += "]";
          if (tensor_list_shape_data[i].data()) {
            result += ", data[";
            for (size_t i = 0; i < tensor_list_shape_data[i].data()->size();
                 ++i) {
              result += ToString(tensor_list_shape_data[i].data()->at(i));
              if (i < tensor_list_shape_data[i].data()->size() - 1) {
                result += ", ";
              }
            }
            result += "]";
          } else {
            result += ", data[NULL]";
          }

          if (i < tensor_list_shape_data.size() - 1) {
            result += ", ";
          }
        }
      }};

  std::visit(lambdas, shape_or_data.variant());

  stream << result;

  return stream;
}

namespace {

std::size_t GetHashValueImpl(const std::int64_t& dim_expr) { return dim_expr; }

std::size_t GetHashValueImpl(const std::string& dim_expr) {
  return std::hash<std::string>()(dim_expr);
}

std::size_t GetHashValueImpl(const Negative<DimExpr>& dim_expr) {
  return -GetHashValue(dim_expr->data);
}

std::size_t GetHashValueImpl(const Reciprocal<DimExpr>& dim_expr) {
  return pir::hash_combine(1, -GetHashValue(dim_expr->data));
}

std::size_t GetHashValueImpl(const List<DimExpr>& exprs) {
  std::size_t ret = 0;
  for (const auto& expr : *exprs) {
    ret = pir::hash_combine(ret, GetHashValue(expr));
  }
  return ret;
}

std::size_t GetHashValueImpl(const Add<DimExpr>& dim_expr) {
  return pir::hash_combine(1, GetHashValueImpl(dim_expr.operands));
}

std::size_t GetHashValueImpl(const Mul<DimExpr>& dim_expr) {
  return pir::hash_combine(2, GetHashValueImpl(dim_expr.operands));
}

std::size_t GetHashValueImpl(const Max<DimExpr>& dim_expr) {
  return pir::hash_combine(3, GetHashValueImpl(dim_expr.operands));
}

std::size_t GetHashValueImpl(const Min<DimExpr>& dim_expr) {
  return pir::hash_combine(4, GetHashValueImpl(dim_expr.operands));
}

std::size_t GetHashValueImpl(const Broadcast<DimExpr>& dim_expr) {
  return pir::hash_combine(5, GetHashValueImpl(dim_expr.operands));
}

}  // namespace

std::size_t GetHashValue(const DimExpr& dim_expr) {
  return std::visit([](const auto& impl) { return GetHashValueImpl(impl); },
                    dim_expr.variant());
}

}  // namespace symbol
