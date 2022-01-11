/**
 * @name Divided by input tensor dims
 * @description Input tensor dims may be zero. If no sufficient check to protect, 
 *              dividing tensor dims could cause a division by zero problem.
 * @kind path-problem
 * @problem.severity error
 * @precision medium
 * @id cpp/divide-tensor-dims
 * @tags reliability
 *       security
 */

import cpp
import DataFlow::PathGraph
import semmle.code.cpp.dataflow.TaintTracking
import semmle.code.cpp.controlflow.Guards


class Tensor extends LocalScopeVariable {
    Tensor() {
        exists(FunctionCall fc | 
            this.getInitializer().getExpr() = fc and
            fc.getTarget().hasName("Input")
        )
        or
        exists(Type t | 
            this.getType() = t and
            this.getType().hasName("const Tensor *")
        )
    }

    FunctionCall numel() {
        exists(FunctionCall fc | 
            fc.getTarget().hasName("numel") and
            fc.getQualifier() = this.getAnAccess() and
            result = fc
        )
    }

    FunctionCall dims() {
        exists(FunctionCall fc | 
            fc.getTarget().hasName("dims") and
            fc.getQualifier() = this.getAnAccess() and
            result = fc
        )
    }
}

class DivisionByDimsConfig extends TaintTracking::Configuration {
    DivisionByDimsConfig() { this = "DivisionByDimsConfig" }

    override predicate isSource(DataFlow::Node source) {
        exists(Tensor tensor |
            source.asExpr().(FunctionCall).getTarget().hasName("Input") and
            tensor.getInitializer().getExpr() = source.asExpr().(FunctionCall) and
            tensor.getFunction().hasName("Compute")
        )
    }

    override predicate isSink(DataFlow::Node sink) {
        exists(DivExpr de | sink.asExpr().getParent+() = de.getRightOperand()) or
        exists(RemExpr re | sink.asExpr().getParent+() = re.getRightOperand())
    }

    override predicate isAdditionalTaintStep(DataFlow::Node pred, DataFlow::Node succ) {
        exists(Tensor tensor |
            pred.asExpr() = tensor.getAnAccess() and
            (tensor.dims() = succ.asExpr() or tensor.numel() = succ.asExpr())
        )
    }

    override predicate isSanitizer(DataFlow::Node barrier) {
        exists(Macro m, MacroInvocation mi, Tensor tensor |
            m.getName().prefix(15) = "PADDLE_ENFORCE_" and
            m = mi.getMacro() and
            tensor.getAnAccess() = barrier.asExpr() and
            mi.getStmt() = tensor.dims().getEnclosingBlock()
        ) or
        exists(ArrayExpr ae | ae.getArrayOffset() = barrier.asExpr()) or
        barrier.asExpr().getEnclosingFunction().hasQualifiedName("paddle::framework::Tensor::Slice") or
        barrier.asExpr().getEnclosingFunction().hasQualifiedName("paddle::operators::StridedNumelCopyWithAxis")
    }
}

from DataFlow::PathNode source, DataFlow::PathNode sink, DivisionByDimsConfig config, Location loc
where config.hasFlowPath(source, sink) and
    loc = sink.getNode().getLocation()
select sink, source, sink, "loc: $@", loc, loc.toString()