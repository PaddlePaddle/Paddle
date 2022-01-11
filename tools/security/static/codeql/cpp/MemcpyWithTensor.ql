/**
 * @name Memcpy dst offset with tensor data
 * @description Function memcpy dst offset may set by calculation of tensor data.
 *              The result would not be checked and may be out of bounds.
 * @kind problem
 * @problem.severity error
 * @precision medium
 * @id cpp/memcpy-dst-with-tensor
 * @tags security
 */

import cpp
import semmle.code.cpp.dataflow.TaintTracking


from FunctionCall memcpy, FunctionCall fc, MulExpr mul, string var_name
where memcpy.getTarget().hasQualifiedName("memcpy")
    and exists(DataFlow::Node source, DataFlow::Node sink |
        source.asExpr() = fc and
        sink.asExpr().getParent+() = mul and
        TaintTracking::localTaint(source, sink) and
        var_name = sink.toString()
    )
    and fc.getTarget().hasName("data")
    and mul.getParent*() = memcpy.getArgument(0)
select memcpy.getLocation(), "memcpy dst offset is affected by " + var_name