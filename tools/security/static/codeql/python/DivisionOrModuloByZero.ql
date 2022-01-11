/**
 * @name Division or modulo by input
 * @description Input value may be zero. If no sufficient check to protect, 
 *              it could cause a division or modulo by zero problem.
 * @kind problem
 * @problem.severity error
 * @precision medium
 * @id py/division-or-modulo-by-zero
 * @tags reliability
 *       security
 */

import python
import semmle.python.dataflow.new.TaintTracking
import semmle.python.ApiGraphs
import semmle.python.Exprs

class DivisionOrModuloByZeroConfiguration extends TaintTracking::Configuration {
  DivisionOrModuloByZeroConfiguration() { this = "DivisionOrModuloByZeroConfiguration" }

    override predicate isSource(DataFlow::Node source) {
      source instanceof DataFlow::ParameterNode
    }

    override predicate isSink(DataFlow::Node sink) {
      exists( BinaryExprNode bnode | 
        bnode.getRight() = sink.asCfgNode() and 
        bnode.getOp().toString() in ["Mod", "FloorDiv", "Div"] and
        not bnode.getLeft().getAChild*().getNode() instanceof StrConst)
    }

    predicate assertSanitizer(DataFlow::Node barrier) {
      exists(Assert art, Compare cmp, IntegerLiteral integer | 
        art.getASubExpression() = cmp and
        cmp.getComparator(0) = integer and
        integer.getValue() = 0 and
        cmp.getOp(0).toString() in ["NotEq", "Gt", "Lt"] and
        barrier.asExpr() = cmp.getLeft()
      )
    }

    predicate forSanitizer(DataFlow::Node barrier) {
        exists(DataFlow::CallCfgNode call, For fr | 
          fr.getASubExpression().getAChildNode*() = call.asExpr() and
          call = API::builtin("range").getACall() and
          barrier = call.getArg(0)
        )
    }

    override predicate isSanitizer(DataFlow::Node barrier) {
      exists(If i, IntegerLiteral integer, Compare cmp| 
        i.getASubExpression().getAChildNode*() = cmp and
        cmp.getComparator(0) = integer and
        integer.getValue() = 0 and
        cmp.getOp(0).toString() in ["LtE", "Eq"]  and
        barrier.asExpr() = cmp.getLeft()) or 
      exists(If i | i.getASubExpression() = barrier.asExpr()) or 
      assertSanitizer(barrier) or
      forSanitizer(barrier)
    } 

}

predicate excludePrivateFunc(DataFlow::ParameterNode source) {
  exists(Function f | 
      not f.getName().regexpMatch("_(.)+") and f.getAnArg() = source.getParameter()
      )
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

from DataFlow::Node source, DataFlow::Node sink, DivisionOrModuloByZeroConfiguration config
where 
    excludePrivateFunc(source) and 
    excludeNotinterestedFiles(source) and
    config.hasFlow(source, sink)
select sink, "integer division or modulo by zero."