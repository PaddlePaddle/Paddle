#ifdef PTEN_TENSOR
#else
#define PTEN_TENSOR

include "paddle/infrt/dialect/pten/infrt_pten_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "paddle/infrt/dialect/infrt_base.td"

def PTEN_DenseTensorDialect : Dialect {
  let name = "pten_dt";

  let description = [{
    The PTEN DenseTensor dialect.
  }];

  let cppNamespace = "::infrt::pten";
}

// PTEN DenseTensor related Op.
class PDT_Op<string mnemonic, list<OpTrait> traits = []> : Op<PTEN_DenseTensorDialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

class CreateDenseTensorOp<string place, string dtype, string layout> 
      : PDT_Op<"create_dense_tensor." # place # "." # dtype # "." # layout, [NoSideEffect]> {
  let arguments = (ins CPU_Allocator:$allocator, I64ArrayAttr:$dims, I64ArrayAttr:$lod);
  let results = (outs TensorType:$output);
}

class FillDenseTensorOp<Attr attr_type, string dtype> :
      PDT_Op<"fill_dense_tensor." # dtype> {
  let arguments = (ins
      TensorType:$input,
      attr_type:$value
  );
  let results = (outs);
}

class CreateCPUAllocatorOp
      : PDT_Op<"create_allocator." # "cpu", [NoSideEffect]> {
  let arguments = (ins);
  let results = (outs CPU_Allocator:$output);
}

class CreateCPUContextOp
      : PDT_Op<"create_context." # "cpu", [NoSideEffect]> {
  let arguments = (ins);
  let results = (outs CPU_Context:$output);
}

def PDT_CreateDenseTensorOp_cpu_f32_nchw : CreateDenseTensorOp<"cpu", "f32", "nchw">;
def PDT_FillDenseTensorOp_f32 : FillDenseTensorOp<F32ArrayAttr, "f32">;
def PDT_CreateAllocatorOp_cpu : CreateCPUAllocatorOp;
def PDT_CreateContextOp_cpu : CreateCPUContextOp;

#endif
