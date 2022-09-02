// This file defines some basic elements of Paddle(alias pd) dialect.
// We learned much from TensorFlow mlir dialect https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td

#ifndef PD_OP_BASE
#define PD_OP_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "paddle/infrt/dialect/infrt/ir/infrt_base.td"

def Paddle_Dialect : Dialect {
  let name = "pd";

  let description = [{
    The PaddlePaddle dialect.

    This dialect contains the PaddlePaddle operators.
  }];
  let cppNamespace = "infrt::pd";
}

class PD_Op<string mnemonic, list<OpTrait> traits = []> :
      Op<Paddle_Dialect, mnemonic, traits>;


class PD_PaddleAttr <string name, string description> :
      Attr<CPred<"$_self.isa<infrt::pd::" # name # "Attr>()">,
          "PaddlePaddle " # description # " attribute">;


//===----------------------------------------------------------------------===//
// PaddlePaddle type definitions
//===----------------------------------------------------------------------===//

def PD_PDDialectType : Type<CPred<"$_self.isa<infrt::pd::PDType>()">, "PaddlePaddle type">;

class PD_PaddleType <string name, string description> :
      Type<CPred<"$_self.isa<infrt::pd::" # name #"Type>()">,
         "Paddle " # description # " type">,
      BuildableType<"getType<infrt::pd::" # name # "Type>()">;

//===----------------------------------------------------------------------===//
// Integer types
def PD_Bool : AnyTypeOf<[I<1>], "bool">;
def PD_Int8 : AnyTypeOf<[I8], "8-bit integer">;
def PD_Int16 : AnyTypeOf<[I16], "16-bit integer">;
def PD_Int32 : AnyTypeOf<[I32], "32-bit integer">;
def PD_Int64 : AnyTypeOf<[I64], "64-bit integer">;

def PD_UInt8 : AnyTypeOf<[UI<8>], "8-bit unsigned integer">;
def PD_UInt16 : AnyTypeOf<[UI<16>], "16-bit unsigned integer">;
def PD_UInt32 : AnyTypeOf<[UI<32>], "32-bit unsigned integer">;
def PD_UInt64 : AnyTypeOf<[UI<64>], "64-bit unsigned integer">;

def PD_SInt : AnyTypeOf<[PD_Int8, PD_Int16, PD_Int32, PD_Int64], "signed integer">;
def PD_UInt : AnyTypeOf<[PD_UInt8, PD_UInt16, PD_UInt32, PD_UInt64], "unsigned integer">;
def PD_Int : AnyTypeOf<[PD_SInt, PD_UInt], "integer">;

// Float types
def PD_Float16 : AnyTypeOf<[F16], "16-bit float">;
def PD_Float32 : AnyTypeOf<[F32], "32-bit float">;
def PD_Float64 : AnyTypeOf<[F64], "64-bit float">;

def PD_Float : AnyTypeOf<[PD_Float16, PD_Float32, PD_Float64], "floating-point">;


// Tensor types

def PD_ElementType : Type<Or<[PD_Float.predicate,
                              PD_Bool.predicate,
                              PD_Int.predicate]>,
                              "pd.dtype">;

// def PD_Tensor : TensorOf<[PD_ElementType]>;
def PD_Tensor1 : TensorOf<[PD_ElementType]>;

def PD_Tensor :  AnyTypeOf<[PD_Tensor1, LoDTensor, DenseTensor],"pd.ttype">;

def PD_Tensor_Array : VectorOf<[PD_Tensor]>;

#endif // PD_OP_BASE
