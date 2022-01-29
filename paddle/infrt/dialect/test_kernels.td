// Operation definitions for testing.

#ifdef TEST_OPS
#else
#define TEST_OPS

include "paddle/infrt/dialect/infrt_base.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for Test dialect ops.
class Test_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<INFRT_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {

  // Each registered op in the Test namespace needs to provide all of a printer,
  // parser and verifier.
  let printer = [{ return infrt::dialect::print(p, *this); }];
  let verifier = [{ return infrt::dialect::verify(*this); }];
  let parser = [{ return infrt::dialect::parse$cppClass(parser, result); }];
}

def BenchmarkOp : Test_Op<"benchmark"> {
  let summary = "benchmark operation";
  let description = [{
     The "infrt.benchmark" operation benchmarks the performance of an MLIR
     region by executing the given MLIR region repeatedly up to the
     `duratino_secs` seconds or `max_count` times. `num_warmup_runs` specifies
     the number of warm up runs to run the given MLIR region before the
     benchmark starts.

     The target MLIR region can take an arbitrary number of arguments and
     should return exactly one value. The arguments for the MLIR region are
     provided as the operands of the infrt.benchmark op.

     Example:
       infrt.benchmark "add.i32"(%c : i32, %d : f32) max_count = 100, duration_secs = 1 {
         // code for benchmarking
         ...
       }

       infrt.benchmark "add.i32"(%c : i32)
         duration_secs = 1,
         max_count = 100,
         num_warmup_runs = 10 {
         // The MLIR code to be benchmarked goes here.
         // The following code benchmarks the infrt.add.i32 kernel.
         %x = infrt.add.i32 %c, %c
         // The benchmarked function needs to return exactly one value.
         infrt.return %x : i32
       }
  }];

  let regions = (region SizedRegion<1>:$region);

  let arguments = (ins
    Variadic<AnyType>,
    I32Attr:$duration_secs,
    I32Attr:$max_count,
    StrAttr:$name,
    DefaultValuedAttr<I32Attr, "1">:$num_warmup_runs
  );

  let results = (outs);
}

#endif  // TEST_OPS
