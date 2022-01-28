#pragma once
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <string>

namespace infrt {
namespace pten {

struct AllocatorTypeStorage : public mlir::TypeStorage {
  AllocatorTypeStorage(const std::string& kind) : kind_(kind) {}

  bool operator==(const std::string& key) const { return key == kind_; }

  static llvm::hash_code hashKey(const std::string& key) {
    return llvm::hash_value(key);
  }

  static AllocatorTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                         const std::string& key) {
    return new (allocator.allocate<AllocatorTypeStorage>())
        AllocatorTypeStorage(key);
  }

 private:
  std::string kind_;
};

class AllocatorType : public mlir::Type::TypeBase<AllocatorType,
                                                  mlir::Type,
                                                  AllocatorTypeStorage> {
 public:
  using Base::Base;
  static AllocatorType get(const std::string& kind);

  static std::string& kind();
};

}  // namespace pten
}  // namespace infrt
