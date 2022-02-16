// Operation definitions for basic kernels.

#ifdef BASIC_OPS
#else
#define BASIC_OPS

include "paddle/infrt/dialect/infrt_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class INFRT_Op<string mnemonic, list<OpTrait> traits = []> : Op<INFRT_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {

  // Each registered op needs to provide all of a printer, parser and verifier.
  let printer = [{ return infrt::dialect::print(p, *this); }];
  let verifier = [{ return infrt::dialect::verify(*this); }];
  let parser = [{ return infrt::dialect::parse$cppClass(parser, result); }];
}

def CallOp : INFRT_Op<"call"> {
  let summary = "call a host operation";
  let description = [{
      The "infrt.call" operation represents a direct call to a function. The operands and result types of the call must match the specified function type.

          %2 = infrt.call @add(%0, %1) : (f32, f32) -> f32
    }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
      mlir::StringRef getCallee() { return callee(); }
      mlir::FunctionType getCalleeType();
    }];
}

class ConstantOp<string suffix, Type baseType, Attr attr>
    : INFRT_Op<"constant." # suffix, [NoSideEffect]> {
  let summary = "constant value constructor in host";

  let arguments = (ins attr:$value);
  let results = (outs baseType);
}

def ConstantI32Op : ConstantOp<"i32", I32, I32Attr>;
def ConstantI64Op : ConstantOp<"i64", I64, I64Attr>;
def ConstantF32Op : ConstantOp<"f32", F32, F32Attr>;
def ConstantF64Op : ConstantOp<"f64", F64, F64Attr>;

def ReturnOp : INFRT_Op<"return", [Terminator]> {
  let summary = "host executor return operation";
  let description = [{
      The "infrt.return" operation represents a return operation within a function.

        func @foo() : (i32, f8) {
        infrt.return %0, %1 : i32, f8
        }
    }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins),
                  [{ build($_builder, $_state, llvm::None); }]>];
}

class AddOp<string suffix, Type type> : INFRT_Op<"add." # suffix, [NoSideEffect]> {
  let summary = "infrt.add operation";
  let description = [{
      An operation that takes two inputs and returns their sum as result.
    }];

  let arguments = (ins type, type);
  let results = (outs type);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def AddI32Op : AddOp<"i32", I32>;
def AddI64Op : AddOp<"i64", I64>;
def AddF32Op : AddOp<"f32", F32>;
def AddF64Op : AddOp<"f64", F64>;

class MulOp<string suffix, Type type> : INFRT_Op<"mul." # suffix, [NoSideEffect]> {
    let summary = "infrt.mul operation";
    let description = [{
        An operation that takes two inputs and returns their mul as result.
    }];

    let arguments = (ins type, type);
let results = (outs type);
let assemblyFormat = "operands attr-dict";
let verifier = ?;
}

def MulI32Op : MulOp<"i32", I32>;
def MulI64Op : MulOp<"i64", I64>;
def MulF32Op : MulOp<"f32", F32>;
def MulF64Op : MulOp<"f64", F64>;

class PrintOp<string suffix, Type type> : INFRT_Op<"print." # suffix> {
  let summary = "infrt.print operation";
  let description = [{
      An operation takes a number as input and prints to stdout.
    }];

  let arguments = (ins type);
  let assemblyFormat = "operands attr-dict";
  let verifier = ?;
}

def PrintI32Op : PrintOp<"i32", I32>;
def PrintI64Op : PrintOp<"i64", I64>;
def PrintF32Op : PrintOp<"f32", F32>;
def PrintF64Op : PrintOp<"f64", F64>;

def GetStringOp : INFRT_Op<"get_string"> {
  let summary = "infrt.get_string";
  let description = [{
    Get a !infrt.string value from the given string attribute.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs StringType);
  let assemblyFormat = "`(` $value `)` attr-dict";
  let verifier = ?;
}

def PrintStringOp : INFRT_Op<"print_string"> {
  let summary = "infrt.print_string";
  let description = [{
      An operation that prints a string.
  }];

  let arguments = (ins StringType:$input);
  let results = (outs);
  let assemblyFormat = "`(` $input `)` attr-dict";
  let verifier = ?;
}

#endif  // basic kernels
