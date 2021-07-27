#ifndef BAIDU_BAIDU_PSLIB_SGD_SPARSE_SGD_H
#define BAIDU_BAIDU_PSLIB_SGD_SPARSE_SGD_H
#include <vector>
#include <thread>
#include <math.h>
#include "glog/logging.h"       // for CHECK
#include "common/local_random.h"    // for local_uniform_real_distribution
#include "Eigen/Dense"
#include "proto/ps.pb.h"

namespace paddle {
namespace ps {

inline std::vector<float>& local_float_vec() {
    thread_local std::vector<float> vec;
    return vec;
}

inline std::vector<float>& local_gradient_vec() {
    thread_local std::vector<float> vec;
    return vec;
}

inline std::vector<float>& local_g2sum_vec() {
    thread_local std::vector<float> vec;
    return vec;
}

inline std::vector<float>& local_score_vec() {
    thread_local std::vector<float> vec;
    return vec;
}

class SparseValueSGDRule {
public:
    virtual ~SparseValueSGDRule() {}
    virtual void load_config(const SparseCommonSGDRuleParameter& param, size_t emb_dim) {
        _embedding_dim = emb_dim;
        _name = param.name();
    }
    virtual void update_value_work(float* w, float* sgd, const float* push_value, float scale) = 0;
    virtual void init_value_work(float* value, float* sgd, bool zero_init) = 0;
    virtual size_t dim() = 0; 
    const std::string& get_name() const {
        return _name;
    }
    void init_value(float* value, float* sgd, bool zero_init = false) {
        init_value_work(value, sgd, zero_init);
    }
    void update_value(float* w, float* sgd, const float* push_value, float scale = 1) {
        update_value_work(w, sgd, push_value, scale);
    }
    template<class T>
    void bound_value(T& w) {
        if (!(w >= _min_bound)) {
            w = (T)_min_bound;
        } else if (!(w <= _max_bound)) {
            w = (T)_max_bound;
        }
    }
    float& min_bound() {
        return _min_bound;
    }
    float& max_bound() {
        return _max_bound;
    }
protected:
    float _min_bound;
    float _max_bound;
    float _initial_range;
    size_t _embedding_dim;
private:
    std::string _name;
};

class SparseNaiveSGDRule : public SparseValueSGDRule {
public:
    virtual void load_config(const SparseCommonSGDRuleParameter& param, size_t emb_dim);
    virtual void update_value_work(float* w, float* sgd, const float* push_value, float scale);
    virtual void init_value_work(float* value, float* sgd, bool zero_init);
    virtual size_t dim() {
        return 0;
    }
private:
    float _learning_rate;
};

class SparseAdaGradSGDRule : public SparseValueSGDRule {
public:
    virtual void load_config(const SparseCommonSGDRuleParameter& param, size_t emb_dim);
    virtual void update_value_work(float* w, float* sgd, const float* push_value, float scale);
    virtual void init_value_work(float* value, float* sgd, bool zero_init);
    virtual size_t dim() {
        return 1;
    }
    size_t g2sum_index() {
        return 0;
    }
private:
    float _learning_rate;
    float _initial_g2sum;
};

class StdAdaGradSGDRule : public SparseValueSGDRule {
public:
    virtual void load_config(const SparseCommonSGDRuleParameter& param, size_t emb_dim);
    virtual void update_value_work(float* w, float* sgd, const float* push_value, float scale);
    virtual void init_value_work(float* value, float* sgd, bool zero_init);
    virtual size_t dim() {
        return _embedding_dim;
    }
    size_t g2sum_index() {
        return 0;
    }
private:
    float _learning_rate;
    float _initial_g2sum;
};

class SparseAdamSGDRule : public SparseValueSGDRule {
public:
    virtual void load_config(const SparseCommonSGDRuleParameter& param, size_t emb_dim);
    virtual void update_value_work(float* w, float* sgd, const float* push_value, float scale);
    virtual void init_value_work(float* value, float* sgd, bool zero_init);
    virtual size_t dim() {
        return _embedding_dim * 2 + 2;
    }
    size_t gsum_index(){
        return 0;
    }
    size_t g2sum_index(){
        return gsum_index() + _embedding_dim;
    }
    size_t beta1_pow_index(){
        return g2sum_index() + _embedding_dim;
    }
    size_t beta2_pow_index(){
        return beta1_pow_index() + 1;
    }
protected:
    float _learning_rate;
    float _beta1_decay_rate;
    float _beta2_decay_rate;
    float _ada_epsilon;
};

struct HitIntervalSGDRule {
    float learning_rate, initial_value;

    void load_config(const HitIntervalSGDRuleParameter& param) {
        learning_rate = param.learning_rate();
        initial_value = param.initial_value();
    }
    template<class T>
    void init_value(int n, T w[], bool zero_intialized = false) {
        for (int i = 0; i < n; i++) {
            w[i] = initial_value;
        }
    }

    template<class T>
    void update_value(int n, T w[], const T hit_interval_new[], const T need_hit_interval[]) {
        for (int i = 0; i < n; i++) {
            if (fabs(need_hit_interval[i] - 1.0) < 1e-4) {
                w[i] = (1 - learning_rate) * w[i] + learning_rate * hit_interval_new[i];
            } else {
                w[i] = 0.0;
            }
        }
    }
};

}
}
#endif
