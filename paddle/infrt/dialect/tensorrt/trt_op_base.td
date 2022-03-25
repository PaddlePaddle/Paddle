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


class TRT_PaddleAttr <string name, string description> :
      Attr<CPred<"$_self.isa<mlir::trt::" # name # "Attr>()">,
          "PaddlePaddle " # description # " attribute">;

def TRT_EngineType :
      Type<CPred<"$_self.isa<::infrt::trt::EngineType>()">, "!trt.engine">,
      BuildableType<"getType<::infrt::trt::EngineType>()">;

//===----------------------------------------------------------------------===//
// PaddlePaddle type definitions
//===----------------------------------------------------------------------===//

def TRT_TRTDialectType : Type<CPred<"$_self.isa<mlir::trt::TRTType>()">, "PaddlePaddle type">;

class TRT_PaddleType <string name, string description> :
      Type<CPred<"$_self.isa<mlir::trt::" # name #"Type>()">,
         "Paddle " # description # " type">,
      BuildableType<"getType<mlir::trt::" # name # "Type>()">;

//===----------------------------------------------------------------------===//
// Integer types
def TRT_Bool : AnyTypeOf<[I<1>], "bool">;
def TRT_Int8 : AnyTypeOf<[I8], "8-bit integer">;
def TRT_Int16 : AnyTypeOf<[I16], "16-bit integer">;
def TRT_Int32 : AnyTypeOf<[I32], "32-bit integer">;
def TRT_Int64 : AnyTypeOf<[I64], "64-bit integer">;

def TRT_UInt8 : AnyTypeOf<[UI<8>], "8-bit unsigned integer">;
def TRT_UInt16 : AnyTypeOf<[UI<16>], "16-bit unsigned integer">;
def TRT_UInt32 : AnyTypeOf<[UI<32>], "32-bit unsigned integer">;
def TRT_UInt64 : AnyTypeOf<[UI<64>], "64-bit unsigned integer">;

def TRT_SInt : AnyTypeOf<[TRT_Int8, TRT_Int16, TRT_Int32, TRT_Int64], "signed integer">;
def TRT_UInt : AnyTypeOf<[TRT_UInt8, TRT_UInt16, TRT_UInt32, TRT_UInt64], "unsigned integer">;
def TRT_Int : AnyTypeOf<[TRT_SInt, TRT_UInt], "integer">;

// Float types
def TRT_Float16 : AnyTypeOf<[F16], "16-bit float">;
def TRT_Float32 : AnyTypeOf<[F32], "32-bit float">;
def TRT_Float64 : AnyTypeOf<[F64], "64-bit float">;

def TRT_Float : AnyTypeOf<[TRT_Float16, TRT_Float32, TRT_Float64], "floating-point">;


// Tensor types

def TRT_ElementType : Type<Or<[TRT_Float.predicate,
                              TRT_Bool.predicate,
                              TRT_Int.predicate]>,
                              "trt.dtype">;

def TRT_Tensor : TensorOf<[TRT_ElementType]>;


#endif // TRT_OP_BASE
