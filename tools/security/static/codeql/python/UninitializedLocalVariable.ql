/**
 * @name Potentially uninitialized local variable in Paddle
 * @description Using a local variable before it is initialized causes an UnboundLocalError.
 * @kind problem
 * @tags reliability
 *       correctness
 * @problem.severity error
 * @sub-severity low
 * @precision high
 * @id py/uninitialized-local-variable
 */

import python
import Variables.Undefined
import Variables.Loop
import semmle.python.pointsto.PointsTo

predicate uninitialized_local(NameNode use) {
  exists(
    FastLocalVariable local | use.uses(local) or use.deletes(local) | not local.escapes()
  ) and (
    any(Uninitialized uninit).taints(use) and
    PointsToInternal::reachableBlock(use.getBasicBlock(), _)
    or not exists(EssaVariable var | var.getASourceUse() = use)
    or (
      exists(FastLocalVariable local | use.uses(local) or use.deletes(local) | 
        probably_defined_in_loop(use.getNode()) and
        not exists(Name s | s = local.getAStore() and not define_in_loop(local).contains(s)) and 
        not define_in_loop(local).contains(use.getNode())
      )
    )
  )
}

For define_in_loop(FastLocalVariable v) {
  result.contains(v.getAStore())
}

predicate explicitly_guarded(NameNode u) {
  exists(Try t |
    t.getBody().contains(u.getNode()) and
    t.getAHandler().getType().pointsTo(ClassValue::nameError())
  )
}

from NameNode u, File f
where uninitialized_local(u) and 
      not explicitly_guarded(u) and 
      u.getLocation().getFile() = f and 
      not f.getShortName().matches("test\\_%.py") and 
      not f.getShortName().matches("%\\_fuzz.py") and 
      not f.getShortName().matches("%\\_test.py")
select u.getNode(), "Local variable '" + u.getId() + "' may be used before it is initialized. " + u.getLocation().toString()