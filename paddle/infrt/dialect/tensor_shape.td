#ifdef INFRT_OPS
#else
#define INFRT_OPS

include "paddle/infrt/dialect/infrt/ir/infrt_base.td"
include "paddle/infrt/dialect/tensor_shape_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in the TensorShape dialect
class TS_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TensorShapeDialect, mnemonic, traits> {
  let parser = [{ return infrt::dialect::parse$cppClass(parser, result); }];
  let printer = " return infrt::dialect::printOpWithOperands(p, *this)" ";";
}

def TS_BuildShapeOp : TS_Op<"build_shape", [NoSideEffect]> {
  let summary = "Build tensor shape operation";
  let description = [{
    An operation that builds a tensor shape of given ranks and extents.
  }];

  let arguments = (ins I64ArrayAttr:$value);
  let results = (outs TS_Shape:$output);
  let assemblyFormat = "$value attr-dict";
}

def TS_GetNumElementsOp : TS_Op<"get_num_elements"> {
  let summary = "Returns the number of elements in the shape";

  let description = [{
    An operation that returns the number of elements in the given shape.
  }];

  let arguments = (ins TS_Shape);
  let results = (outs I64);
  let assemblyFormat = "operands attr-dict";
}

def TS_PrintShapeOp : TS_Op<"print_shape"> {
  let summary = "Print tensor shape operation";
  let description = [{
    An operation that prints a tensor shape.
  }];

  let arguments = (ins TS_Shape:$shape);
  let assemblyFormat = "operands attr-dict";
}

#endif
