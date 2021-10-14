#ifndef BAIDU_BAIDU_PSLIB_INCLUDE_SGD_SPARSE_VALUE_SGD_FACTORY_H
#define BAIDU_BAIDU_PSLIB_INCLUDE_SGD_SPARSE_VALUE_SGD_FACTORY_H
#include "common/factory.h"
#include "sparse_sgd.h"
namespace paddle {
namespace ps {
    
inline Factory<SparseValueSGDRule>& global_sparse_value_sgd_rule_factory() {
    static Factory<SparseValueSGDRule> f;
    return f;
}

inline void pslib_sparse_sgd_init() {
    static bool sgd_initial = false;

    if (sgd_initial) {
        return;
    }
    sgd_initial = true;
    
    Factory<SparseValueSGDRule>& factory = global_sparse_value_sgd_rule_factory();
    factory.add<SparseNaiveSGDRule>("naive");
    factory.add<SparseAdaGradSGDRule>("adagrad");
    factory.add<StdAdaGradSGDRule>("std_adagrad");
    factory.add<SparseAdamSGDRule>("adam");
}

}
}
#endif
