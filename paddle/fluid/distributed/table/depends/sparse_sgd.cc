#include "sgd/sparse_sgd.h"
#include <gflags/gflags.h>

DEFINE_bool(enable_show_scale_gradient, true, "enable show scale gradient");

namespace paddle {
namespace ps {

void SparseSGDRule::update_value(int row, int col, float** w, float* g2sum, const float** grad, const float* g_scale) {
    static bool show_scale = FLAGS_enable_show_scale_gradient;

    Eigen::Map<const Eigen::MatrixXf> mat_g_scale(g_scale, 1, col);

    local_float_vec().resize(col, 0.0);

    Eigen::Map<Eigen::MatrixXf> mat_add_g2sum(local_float_vec().data(), 1, col);
    mat_add_g2sum = Eigen::MatrixXf::Zero(1, col);

    local_g2sum_vec().resize(col, 0.0);
    memcpy(local_g2sum_vec().data(), g2sum, sizeof(float) * col);

    Eigen::Map<Eigen::MatrixXf> mat_g2sum(local_g2sum_vec().data(), 1, col);

    mat_g2sum = ((mat_g2sum.array() + initial_g2sum).cwiseInverse() * initial_g2sum).cwiseSqrt() * learning_rate;

    local_gradient_vec().resize(col);

    for (auto i = 0u; i < row; ++i) {
        Eigen::Map<Eigen::MatrixXf> mat_w(w[i], 1, col);

        memcpy(local_gradient_vec().data(), grad[i], sizeof(float) * col);

        Eigen::Map<Eigen::MatrixXf> mat_grad(local_gradient_vec().data(), 1, col);

        if (show_scale) {
            mat_grad = mat_grad.cwiseQuotient(mat_g_scale);
        }

        mat_w -= mat_grad.cwiseProduct(mat_g2sum);

        mat_add_g2sum += mat_grad.cwiseProduct(mat_grad);
    }

    Eigen::Map<Eigen::MatrixXf> output_mat_g2sum(g2sum, 1, col);
    output_mat_g2sum += mat_add_g2sum / row;

    bound_value(row, col, w);
}

void SparseNaiveSGDRule::load_config(const SparseCommonSGDRuleParameter& param, size_t emb_dim) {
    _embedding_dim = emb_dim;
    auto naive_param = param.naive();
    _learning_rate = naive_param.learning_rate();
    _initial_range = naive_param.initial_range();
    if (naive_param.weight_bounds_size() == 0) {
        _min_bound = -std::numeric_limits<float>::max();
        _max_bound = std::numeric_limits<float>::max();
    } else {
        CHECK(naive_param.weight_bounds_size() >= 2) 
            << "invalid repeated size for weight_bounds:" << naive_param.weight_bounds_size();
        _min_bound = naive_param.weight_bounds(0);
        _max_bound = naive_param.weight_bounds(1);
    }
}

void SparseNaiveSGDRule::update_value_work(float* w, float* sgd, const float* push_value, float scale) {
    for (size_t i = 0; i < _embedding_dim; ++i) {
        w[i] -= _learning_rate * push_value[i];
        bound_value(w[i]);
    }
}

void SparseNaiveSGDRule::init_value_work(float* value, float* sgd, bool zero_init) {
    if (zero_init) {
        for (size_t i = 0; i < _embedding_dim; ++i) {
            value[i] = 0;
        }
    } else {
        for (size_t i = 0 ; i < _embedding_dim; ++i) {
            value[i] = (local_uniform_real_distribution<float>()(
                        local_random_engine()) * 2 - 1) * _initial_range;
            bound_value(value[i]);
        }
    }
}
void SparseAdaGradSGDRule::load_config(const SparseCommonSGDRuleParameter& param, size_t emb_dim) {
    _embedding_dim = emb_dim;
    auto adagrad_param = param.adagrad();
    _learning_rate = adagrad_param.learning_rate();
    _initial_g2sum = adagrad_param.initial_g2sum();
    _initial_range = adagrad_param.initial_range();

    if (adagrad_param.weight_bounds_size() == 0) {
        _min_bound = -std::numeric_limits<float>::max();
        _max_bound = std::numeric_limits<float>::max();
    } else {
        CHECK(adagrad_param.weight_bounds_size() >= 2) 
            << "invalid repeated size for weight_bounds:" << adagrad_param.weight_bounds_size();
        _min_bound = adagrad_param.weight_bounds(0);
        _max_bound = adagrad_param.weight_bounds(1);
    }
}

void SparseAdaGradSGDRule::update_value_work(float* w, float* sgd, const float* grad, float scale) {
    float& g2sum = sgd[g2sum_index()];
    double add_g2sum = 0;

    for (int i = 0; i < _embedding_dim; i++) {
        double scaled_grad = grad[i] / scale;
        w[i] -= _learning_rate * scaled_grad * sqrt(_initial_g2sum / (_initial_g2sum + g2sum));
        bound_value(w[i]);
        add_g2sum += scaled_grad * scaled_grad;
    }

    g2sum += add_g2sum / _embedding_dim;
}

void SparseAdaGradSGDRule::init_value_work(float* value, float* sgd, bool zero_init) {
    for (int i = 0; i < _embedding_dim; ++i) {
        if (zero_init) {
            value[i] = 0.0;
            bound_value(value[i]);
        }
        else {
            value[i] = (local_uniform_real_distribution<double>()(
                        local_random_engine()) * 2 - 1) * _initial_range;
            bound_value(value[i]);
        }
    }
    sgd[g2sum_index()] = 0;
}

void StdAdaGradSGDRule::load_config(const SparseCommonSGDRuleParameter& param, size_t emb_dim) {
    _embedding_dim = emb_dim;
    auto adagrad_param = param.adagrad();
    _learning_rate = adagrad_param.learning_rate();
    _initial_g2sum = adagrad_param.initial_g2sum();
    _initial_range = adagrad_param.initial_range();

    if (adagrad_param.weight_bounds_size() == 0) {
        _min_bound = -std::numeric_limits<float>::max();
        _max_bound = std::numeric_limits<float>::max();
    } else {
        CHECK(adagrad_param.weight_bounds_size() >= 2) 
            << "invalid repeated size for weight_bounds:" << adagrad_param.weight_bounds_size();
        _min_bound = adagrad_param.weight_bounds(0);
        _max_bound = adagrad_param.weight_bounds(1);
    }
}

void StdAdaGradSGDRule::update_value_work(float* w, float* sgd, const float* grad, float scale) {
    for (int i = 0; i < _embedding_dim; i++) {
        float& g2sum = sgd[g2sum_index() + i];
        double scaled_grad = grad[i] / scale;
        w[i] -= _learning_rate * scaled_grad * sqrt(_initial_g2sum / (_initial_g2sum + g2sum));
        bound_value(w[i]);
        g2sum += scaled_grad * scaled_grad;
    }
}

void StdAdaGradSGDRule::init_value_work(float* value, float* sgd, bool zero_init) {
    for (int i = 0; i < _embedding_dim; ++i) {
        if (zero_init) {
            value[i] = 0.0;
            bound_value(value[i]);
        }
        else {
            value[i] = (local_uniform_real_distribution<double>()(
                        local_random_engine()) * 2 - 1) * _initial_range;
            bound_value(value[i]);
        }
        sgd[g2sum_index() + i] = 0;
    }
}


void SparseAdamSGDRule::load_config(const SparseCommonSGDRuleParameter& param, size_t emb_dim) {
    _embedding_dim = emb_dim;
    auto adam_param = param.adam();
    _learning_rate = adam_param.learning_rate();
    _initial_range = adam_param.initial_range();
    _beta1_decay_rate = adam_param.beta1_decay_rate();
    _beta2_decay_rate = adam_param.beta2_decay_rate();
    _ada_epsilon = adam_param.ada_epsilon();
    if (adam_param.weight_bounds_size() == 0) {
        _min_bound = -std::numeric_limits<float>::max();
        _max_bound = std::numeric_limits<float>::max();
    } else {
        CHECK(adam_param.weight_bounds_size() >= 2) 
            << "invalid repeated size for weight_bounds:" << adam_param.weight_bounds_size();
        _min_bound = adam_param.weight_bounds(0);
        _max_bound = adam_param.weight_bounds(1);
    }
}

void SparseAdamSGDRule::update_value_work(float* w, float* sgd, const float* grad, float scale) {
        float* gsum = sgd + gsum_index();
        float* g2sum = sgd + g2sum_index();
        float* beta1_pow = sgd + beta1_pow_index();
        float* beta2_pow = sgd + beta2_pow_index();
        const float* g = grad;

        float lr = _learning_rate;
        float beta1_pow_ = *beta1_pow;
        float beta2_pow_ = *beta2_pow;

        // lr not change in one update
        lr *= sqrt(1 - beta2_pow_) / (1 - beta1_pow_);
        for (int i = 0; i < _embedding_dim; i++){
            // Calculation
            gsum[i] = _beta1_decay_rate * gsum[i] + (1 - _beta1_decay_rate) * g[i];
            g2sum[i] = _beta2_decay_rate * g2sum[i] + (1 - _beta2_decay_rate) * g[i] * g[i];
            w[i] = w[i] - lr * (gsum[i] / (sqrt(g2sum[i]) + _ada_epsilon));
            bound_value(w[i]);
        }
        // update beta_pow_decay
        (*beta1_pow) *= _beta1_decay_rate;
        (*beta2_pow) *= _beta2_decay_rate;
}

void SparseAdamSGDRule::init_value_work(float* value, float* sgd, bool zero_init) {
    for (int i = 0; i < _embedding_dim; ++i) {
        if (zero_init) {
            value[i] = 0.0;
            bound_value(value[i]);
        }
        else {
            value[i] = (local_uniform_real_distribution<double>()(
                        local_random_engine()) * 2 - 1) * _initial_range;
            bound_value(value[i]);
        }
    }
    // init rule gsum and g2sum
    for (int i = gsum_index(); i < beta1_pow_index(); i++) {
        sgd[i] = 0.0;
    }
    // init beta1_pow and beta2_pow
    *(sgd + beta1_pow_index()) = _beta1_decay_rate;
    *(sgd + beta2_pow_index()) = _beta2_decay_rate;
}


}
}
