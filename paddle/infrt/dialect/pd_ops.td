#ifndef PD_OPS
#define PD_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/pd_op_base.td"

def PD_FeedOp : PD_Op<"feed", [NoSideEffect]> {
  let summary = "Feed Op";

  let description = [{
    Feed a tensor into the model.
  }];

  let arguments = (ins);
  let results = (outs PD_Tensor:$out);

  let assemblyFormat = [{
      `(` `)` attr-dict `:` type($out)
  }];
}

def PD_FetchOp : PD_Op<"fetch", [Terminator]> {
  let summary = "fetch Op";

  let description = [{
    Fetch tensor from the graph.
  }];

  let arguments = (ins Variadic<PD_Tensor>:$inputs);
}

def PD_GraphOp : PD_Op<"graph", [SingleBlockImplicitTerminator<"FetchOp">]> {
  let summary = "paddle graph Op";
  let description = [{
    Describe a paddle graph or subgraph.
  }];
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins Variadic<PD_Tensor>:$inputs);
  let results = (outs Variadic<PD_Tensor>:$outputs);
}

def PD_ConstantOp : PD_Op<"constant", [NoSideEffect, ConstantLike, DeclareOpInterfaceMethods<InferTypeOpInterface>, AllTypesMatch<["value", "output"]>]> {
  let summary = "constant Op";
  let description = [{}];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs PD_Tensor:$output);
  let hasFolder = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Attribute value">,
  ];
}

def PD_AbsOp : PD_Op<"abs", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the absolute value of a tensor";

  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x);
  let results = (outs PD_Tensor:$y);
}

def PD_SqrtOp : PD_Op<"sqrt", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the sqrt value of a tensor";

  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x);
  let results = (outs PD_Tensor:$y);
}

def PD_ReluOp : PD_Op<"relu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the Relu of a tensor";

  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x);
  let results = (outs PD_Tensor:$y);
  let hasCanonicalizer = 1;
}

def PD_Relu6Op : PD_Op<"relu6", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Computes the Relu6 of a tensor";

  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x);
  let results = (outs PD_Tensor:$y);
}

def PD_ElementwiseAdd : PD_Op<"elementwise_add", [NoSideEffect, Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "ElementwiseAdd Op";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y, DefaultValuedAttr<I32Attr, "-1">:$axis);
  let results = (outs PD_Tensor:$out);
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def PD_ElementwiseSub : PD_Op<"elementwise_sub", [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "ElementwiseSub Op";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y, DefaultValuedAttr<I32Attr, "-1">:$axis);
  let results = (outs PD_Tensor:$out);
}

def PD_ElementwiseMul : PD_Op<"elementwise_mul", [NoSideEffect, Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "ElementwiseMul Op";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y, DefaultValuedAttr<I32Attr, "-1">:$axis);
  let results = (outs PD_Tensor:$out);
}

def PD_ElementwiseDiv : PD_Op<"elementwise_div", [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "ElementwiseDiv Op";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y, DefaultValuedAttr<I32Attr, "-1">:$axis);
  let results = (outs PD_Tensor:$out);
}

def PD_MatmulOp : PD_Op<"matmul", [NoSideEffect]> {
  let summary = "Computes the matrix mulplication result of two tensors";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y,
                  DefaultValuedAttr<BoolAttr, "false">:$transpose_x,
                  DefaultValuedAttr<BoolAttr, "false">:$transpose_y,
                  DefaultValuedAttr<F32Attr, "1.0">:$alpha);
  let results = (outs PD_Tensor:$out);

  //let hasCanonicalizer = 1;
}

def PD_MulOp : PD_Op<"mul", [NoSideEffect, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "paddle mul op";
  let description = [{}];

  let arguments = (ins PD_Tensor:$x, PD_Tensor:$y);
  let results = (outs PD_Tensor:$out);

  //let hasCanonicalizer = 1;
}

def PD_Conv2dOp : PD_Op<"conv2d", [NoSideEffect]> {
  let summary = "paddle conv2d operation";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$Input, PD_Tensor:$Filter, PD_Tensor:$Bias);
  let results = (outs PD_Tensor:$Output);

  //let hasCanonicalizer = 1;
}

def PD_BatchNormOp : PD_Op<"batch_norm", [NoSideEffect]> {
  let summary = "paddle batch_norm operation";
  let description = [{
  }];

  let arguments = (ins PD_Tensor:$X, PD_Tensor:$Scale, PD_Tensor:$Bias,
                   PD_Tensor:$Mean, PD_Tensor:$Variance,
                   DefaultValuedAttr<F32Attr, "1e-05">:$epsilon);
  let results = (outs PD_Tensor:$Y);

  let hasCanonicalizer = 1;
}

def PD_FusedFC : PD_Op<"fc", [NoSideEffect]> {
    let summary = "Computes the Fully Connected result of two tensors";
    let description = [{
    }];

    let arguments = (ins PD_Tensor:$input, PD_Tensor:$w, PD_Tensor:$bias, DefaultValuedAttr<I32Attr, "1">:$in_num_col_dims);
    let results = (outs PD_Tensor:$out);
}

def PD_FusedRepeatedFCRelu : PD_Op<"fusion_repeated_fc_relu", [SameVariadicOperandSize, NoSideEffect]> {
    let summary = "";
    let description = [{ }];

    let arguments = (ins PD_Tensor:$input, Variadic<PD_Tensor>:$w, Variadic<PD_Tensor>:$bias);
    let results = (outs PD_Tensor:$out);
    let hasCanonicalizer = 1;
}

#endif  // PD_OPS
