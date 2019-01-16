#pragma once

#include <boost/variant.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace paddle {
namespace inference {
namespace anakin {

enum DataType { kUnk = -1, kFloat32, kFloat64, kInt32 };
enum Place { kCpu = 0, kGpu };

using shape_t = std::vector<int>;
using attr_t = boost::variant<int, bool, float>;

struct Tensor {
  // Resize the shape of this tensor if needed, used by setting input or output.
  void Resize(const shape_t& shape);

  void SetName(const std::string& name);

  const std::string& name() const;

  // Get the tensor's data lazily, will re-malloc if the memory required (shape
  // * dtype) is changed.
  template <typename T>
  T* mutable_data(Place place);

  // Get the tensor's data.
  template <typename T>
  T* data(Place* place, size_t* size) const;

  // Get the datatype of this tensor.
  DataType dtype() const;

  const shape_t& shape() const;

 private:
  std::string name_;
  shape_t shape_;
  void* data_;
};

class AnakinEngine {
 public:
  using attrs_t = std::map<std::string, attr_t>;
  // Is this op is supported by the engine.
  // TODO is the inputs and outputs of this op needed?
  static bool IsOpSupported(const std::string& op_type, const attrs_t& attrs);

  // @param tensor the tensor with Resize callback.
  void DeclareInput(const std::string& id, const Tensor* tensor,
                    const attrs_t& attrs);
  void DeclareOutput(const std::string& id, const Tensor* tensor,
                     const attrs_t& attrs);

  void AddOp(const std::string& op_type, const std::vector<std::string>& inputs,
             const std::vector<std::string>& outputs, const attrs_t& attrs);

  void AddVar(const std::string& id, DataType dtype, const shape_t& shape);

  Tensor* AddWeight(const std::string& id, const Tensor& v);

  std::unique_ptr<AnakinEngine> Clone();

  // Finish creating the network.
  void FreezeNetwork();

  // Execute the network, and the engine will read the declared inputs and write
  // result to the declared outputs.
  void Execute(int batch_size = 0);

 private:
  void* raw_engine_{nullptr};
};
