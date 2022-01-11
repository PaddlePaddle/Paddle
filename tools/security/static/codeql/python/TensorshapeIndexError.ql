/**
 * @name visit a empty tensor shape
 * @description The shape of input tensor may be empty. If no sufficient check to protect, 
 *              input.shape[index] would raise an IndexError and 
 *              input.shape[index:] would raise a ValueError.
 * @kind problem
 * @problem.severity error
 * @precision medium
 * @id py/tensor_shape_indexerror
 * @tags reliability
 *       security
 */

import python
import semmle.python.dataflow.new.TaintTracking
import semmle.python.ApiGraphs

predicate getCallName(Value val, CallNode call, string funame) {
    exists( | val.getACall() = call and val.getName() = funame)
}

predicate raiseSanitizer(DataFlow::ParameterNode source) {
    not exists(If i, RaiseStmtNode raise, Value errorval, CallNode errorcall, Value lenval,
        CallNode lencall, DataFlow::Node sink | 
        i.getASubStatement() = raise.getNode() and 
        raise.getException() = errorcall and 
        getCallName(errorval, errorcall, "ValueError") and i.getAChildNode+() = lencall.getNode() and 
        getCallName(lenval, lencall, "len") and 
        sink.asCfgNode() = i.getASubExpression().getAChildNode+().getAFlowNode() | 
        TaintTracking::localTaint(source, sink))
}

predicate assertSanitizer(DataFlow::ParameterNode source) {
    not exists( Assert art, Value len, CallNode lencall, DataFlow::Node sink | 
        art.getAChildNode+() = lencall.getNode() and
        getCallName(len, lencall, "len") and
        sink.asCfgNode() = art.getASubExpression().getAChildNode+().getAFlowNode() |
        TaintTracking::localTaint(source, sink) )
}

predicate excludeNotinterestedFiles(DataFlow::ParameterNode source) {
    exists(File f | source.getLocation().getFile() = f and 
        not f.getShortName().matches("test\\_%.py") and 
        not f.getShortName().matches("%\\_fuzz.py") and 
        not f.getShortName().matches("%\\_test.py") and
        not f.getAbsolutePath().matches("%test%") and
        not f.getAbsolutePath().matches("%numpy%") and
        not f.getAbsolutePath().matches("%paddle/fluid/%") and
        not f.getAbsolutePath().matches("%/Lib/%"))     
  }

predicate excludeShapeIndexZero(SubscriptNode subnode) {
    not exists(IntegerLiteral integer | 
        subnode.getIndex() = integer.getAFlowNode() and integer.getValue() = 0) and
    not exists(IntegerLiteral neginteger, UnaryExpr negative |
            subnode.getIndex() = negative.getAFlowNode() and
            negative.getOp().toString() = "USub" and
            negative.getOperand() = neginteger and 
            neginteger.getValue() = 1
        )
}

predicate excludePrivateFunc(DataFlow::ParameterNode source) {
    exists(Function f | 
        not f.getName().regexpMatch("_(.)+") and f.getAnArg() = source.getParameter()
        )
}

from DataFlow::ParameterNode source, DataFlow::Node sink, AttrNode attr, SubscriptNode subnode
where
    source.getParameter().asName().toString() != "self" and
    sink.asCfgNode() = attr.getObject() and
    attr.getName() = "shape" and
    subnode.getObject() = attr and
    raiseSanitizer(source) and
    assertSanitizer(source) and
    excludeNotinterestedFiles(source) and
    excludeShapeIndexZero(subnode) and
    excludePrivateFunc(source) and 
    TaintTracking::localTaint(source, sink)
select attr, "input.shape, which may be empty, is visited."