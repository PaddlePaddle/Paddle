# 1 "test.cc"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 1 "<command-line>" 2
# 1 "test.cc"

namespace paddle {
namespace distributed {
namespace auto_parallel {
# 15 "test.cc"
STATIC_ASSERT_GLOBAL_NAMESPACE( __reg_spmd_rule_matmul, "REGISTER_SPMD_RULE must be called in global namespace"); int __holder_matmul = ::paddle::distributed::auto_parallel::SPMDRuleMap::Instance().Insert( "matmul", std::make_unique<paddle::distributed::auto_parallel::MatmulSPMDRule>());

}
}
}
