  common::TopoWalker<const StmtPattern*> MakeTopoWalker(
      const OpTopo& op_topo, const std::vector<StmtPattern>& stmt_patterns);

  std::function<bool(const pir::Operation*)> MakePredicatorIsInjectiveSource(
    const OpTopo& op_topo);
