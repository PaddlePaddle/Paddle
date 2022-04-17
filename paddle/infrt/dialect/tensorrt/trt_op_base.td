// This file defines some basic elements of Paddle(alias trt) dialect.
// We learned much from TensorFlow mlir dialect https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td

#ifndef TRT_OP_BASE
#define TRT_OP_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TRT_Dialect : Dialect {
  let name = "trt";

  let description = [{
    The PaddlePaddle dialect.

    This dialect contains the PaddlePaddle operators.
  }];

  let cppNamespace = "::infrt::trt";
}

class TRT_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<TRT_Dialect, mnemonic, traits>;

def TRT_EngineType :
      Type<CPred<"$_self.isa<::infrt::trt::EngineType>()">, "!trt.engine">,
      BuildableType<"getType<::infrt::trt::EngineType>()">;

#endif // TRT_OP_BASE
