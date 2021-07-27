#include "downpour_accessor.h"
#include <gflags/gflags.h>
#include "ps_instance.h"
#include "glog/logging.h"
#include "sgd/dense_sgd.h"
#include "sgd/dense_sgd_factory.h"
#include "sgd/sparse_sgd_factory.h"
#include "common/ps_string.h"
#include "common/common.h"
//TODO: 上面一坨引用

namespace paddle {
namespace ps {

// double unit
int DownpourDoubleUnitAccessor::initialize() {
    auto name = _config.embed_sgd_param().name();
    _embed_sgd_rule = global_sparse_value_sgd_rule_factory().produce(name);
    _embed_sgd_rule->load_config(_config.embed_sgd_param(), 1);

    name = _config.embedx_sgd_param().name();
    _embedx_sgd_rule = global_sparse_value_sgd_rule_factory().produce(name);
    _embedx_sgd_rule->load_config(_config.embedx_sgd_param(), _config.embedx_dim());
   
    unit_feature_value.embed_sgd_dim = _embed_sgd_rule->dim();
    unit_feature_value.embedx_dim = _config.embedx_dim();
    unit_feature_value.embedx_sgd_dim = _embedx_sgd_rule->dim();
    _show_click_decay_rate = _config.downpour_accessor_param().show_click_decay_rate();
    
    return 0;
}

size_t DownpourDoubleUnitAccessor::dim() {
    return unit_feature_value.dim();
}
size_t DownpourDoubleUnitAccessor::dim_size(size_t dim) {
    auto embedx_dim = _config.embedx_dim();
    return unit_feature_value.dim_size(dim, embedx_dim);
}
size_t DownpourDoubleUnitAccessor::size() {
    return unit_feature_value.size();
}
size_t DownpourDoubleUnitAccessor::mf_size() {
    return (_config.embedx_dim() + unit_feature_value.embedx_sgd_dim) * sizeof(float);//embedx embedx_g2sum
}
// pull value
size_t DownpourDoubleUnitAccessor::select_dim() {
    auto embedx_dim = _config.embedx_dim();
    return 3 + embedx_dim;
}
size_t DownpourDoubleUnitAccessor::select_dim_size(size_t dim) {
    return sizeof(float);
}
size_t DownpourDoubleUnitAccessor::select_size() {
    return select_dim() * sizeof(float);
}
// push value
size_t DownpourDoubleUnitAccessor::update_dim() {
    auto embedx_dim = _config.embedx_dim();
    return 4 + embedx_dim;
}
size_t DownpourDoubleUnitAccessor::update_dim_size(size_t dim) {
    return sizeof(float);
}
size_t DownpourDoubleUnitAccessor::update_size() {
    return update_dim() * sizeof(float);
}
bool DownpourDoubleUnitAccessor::shrink(float* value) {
    //auto base_threshold = _config.downpour_accessor_param().base_threshold();
    //auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    //auto delete_threshold = _config.downpour_accessor_param().delete_threshold();
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delete_after_unseen_days = _config.downpour_accessor_param().delete_after_unseen_days();
    auto delete_threshold = _config.downpour_accessor_param().delete_threshold();
    // time_decay first
    unit_feature_value.show(value) *= _show_click_decay_rate;
    unit_feature_value.click(value) *= _show_click_decay_rate;
    // shrink after
    auto score = show_click_score(unit_feature_value.show(value), unit_feature_value.click(value));
    auto unseen_days = unit_feature_value.unseen_days(value);
    if (score < delete_threshold || unseen_days > delete_after_unseen_days) {
        return true;
    }
    return false;
}
bool DownpourDoubleUnitAccessor::save_ssd(float* value) {
    if (unit_feature_value.unseen_days(value) > _ssd_unseenday_threshold) {
        return true;
    }
    return false;
}
bool DownpourDoubleUnitAccessor::save_cache(
        float* value, int param, double global_cache_threshold) {
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    if (show_click_score(unit_feature_value.show(value), unit_feature_value.click(value)) >= base_threshold
        && unit_feature_value.unseen_days(value) <= delta_keep_days) {
        return unit_feature_value.show(value) > global_cache_threshold;
    }
    return false;
}
bool DownpourDoubleUnitAccessor::save(float* value, int param) {
    //auto base_threshold = _config.downpour_accessor_param().base_threshold();
    //auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    //auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    if (param == 2) {
        delta_threshold = 0;
    }
    switch (param) {
        // save all
        case 0:
        {
            return true;
        }
            // save xbox delta
        case 1:
            // save xbox base
        case 2:
        {
            if (show_click_score(unit_feature_value.show(value), unit_feature_value.click(value)) >= base_threshold
                && unit_feature_value.delta_score(value) >= delta_threshold
                && unit_feature_value.unseen_days(value) <= delta_keep_days) {
                //do this after save, because it must not be modified when retry
                if (param == 2) {
                    unit_feature_value.delta_score(value) = 0;
                }
                return true;
            } else {
                return false;
            }
        }
            // already decayed in shrink
        case 3:
        {
            //DownpourCtrFeatureValue::show(value) *= _show_click_decay_rate;
            //DownpourCtrFeatureValue::click(value) *= _show_click_decay_rate;
            //do this after save, because it must not be modified when retry
            //DownpourDoubleUnitFeatureValue::unseen_days(value)++;
            return true;
        }
        // save revert batch_model
        case 5:
            {
                return true;
            }
        default:
            return true;
    };
}

void DownpourDoubleUnitAccessor::update_stat_after_save(float* value, int param) {
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    if (param == 2) {
        delta_threshold = 0;
    }
    switch (param) {
        case 1:
            {
                if (show_click_score(unit_feature_value.show(value), unit_feature_value.click(value)) >= base_threshold
                        && unit_feature_value.delta_score(value) >= delta_threshold
                        && unit_feature_value.unseen_days(value) <= delta_keep_days) {
                    unit_feature_value.delta_score(value) = 0;
                }
            }
            return;
         case 3:
            {
                unit_feature_value.unseen_days(value)++;
            }
            return;
         default:
            return;
    };
}

int32_t DownpourDoubleUnitAccessor::create(float** values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* value = values[value_item];
        value[unit_feature_value.unseen_days_index()] = 0;
        value[unit_feature_value.delta_score_index()] = 0;
        *(double*)(value + unit_feature_value.show_index()) = 0;
        *(double*)(value + unit_feature_value.click_index()) = 0;
        value[unit_feature_value.slot_index()] = -1;
        _embed_sgd_rule->init_value(
                                   value + unit_feature_value.embed_w_index(),
                                   value + unit_feature_value.embed_g2sum_index(), true);
        _embedx_sgd_rule->init_value(
                                    value + unit_feature_value.embedx_w_index(),
                                    value + unit_feature_value.embedx_g2sum_index());
    }
    return 0;
}
bool DownpourDoubleUnitAccessor::need_extend_mf(float* value) {
    auto show = ((double*)(value + unit_feature_value.show_index()))[0];
    auto click = ((double*)(value + unit_feature_value.click_index()))[0];
    //float score = (show - click) * _config.downpour_accessor_param().nonclk_coeff()
    auto score = (show - click) * _config.downpour_accessor_param().nonclk_coeff()
                  + click * _config.downpour_accessor_param().click_coeff();
    //+ click * _config.downpour_accessor_param().click_coeff();
    return score >= _config.embedx_threshold();
}
// from DownpourCtrFeatureValue to DownpourCtrPullValue
int32_t DownpourDoubleUnitAccessor::select(float** select_values, const float** values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* select_value = select_values[value_item];
        float* value = const_cast<float*>(values[value_item]);
        select_value[DownpourDoubleUnitPullValue::show_index()] = (float)*(double*)(value + unit_feature_value.show_index());
        select_value[DownpourDoubleUnitPullValue::click_index()] = (float)*(double*)(value + unit_feature_value.click_index());
        select_value[DownpourDoubleUnitPullValue::embed_w_index()] = value[unit_feature_value.embed_w_index()];
        memcpy(select_value + DownpourDoubleUnitPullValue::embedx_w_index(),
               value + unit_feature_value.embedx_w_index(), embedx_dim * sizeof(float));
    }
    return 0;
}
// from DownpourCtrPushValue to DownpourCtrPushValue
// first dim: item
// second dim: field num
int32_t DownpourDoubleUnitAccessor::merge(float** update_values, const float** other_update_values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    size_t total_dim = DownpourDoubleUnitPushValue::dim(embedx_dim);
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* update_value = update_values[value_item];
        const float* other_update_value = other_update_values[value_item];
        /**(double*)(update_value + DownpourDoubleUnitPushValue::show_index()) += *(double*)(other_update_value + DownpourDoubleUnitPushValue::show_index());
        *(double*)(update_value + DownpourDoubleUnitPushValue::click_index()) += *(double*)(other_update_value + DownpourDoubleUnitPushValue::click_index());
        for (auto i = 3u; i < total_dim; ++i) {
            update_value[i] += other_update_value[i];
        }*/
        for (auto i = 0u; i < total_dim; ++i) {
            if (i != DownpourDoubleUnitPushValue::slot_index()) {
                update_value[i] += other_update_value[i];
            }
        }
    }
    return 0;
}
// from DownpourCtrPushValue to DownpourCtrFeatureValue
// first dim: item
// second dim: field num
int32_t DownpourDoubleUnitAccessor::update(float** update_values, const float** push_values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* update_value = update_values[value_item];
        const float* push_value = push_values[value_item];
        float push_show = push_value[DownpourDoubleUnitPushValue::show_index()];
        float push_click = push_value[DownpourDoubleUnitPushValue::click_index()];
        float slot = push_value[DownpourDoubleUnitPushValue::slot_index()];
        *(double*)(update_value + unit_feature_value.show_index()) += (double)push_show;
        *(double*)(update_value + unit_feature_value.click_index()) += (double)push_click;
        update_value[unit_feature_value.slot_index()] = slot;
        update_value[unit_feature_value.delta_score_index()] +=
                (push_show - push_click) * _config.downpour_accessor_param().nonclk_coeff() +
                push_click * _config.downpour_accessor_param().click_coeff();
        //(push_show - push_click) * _config.downpour_accessor_param().nonclk_coeff() +
        //push_click * _config.downpour_accessor_param().click_coeff();
        update_value[unit_feature_value.unseen_days_index()] = 0;
        _embed_sgd_rule->update_value(
                                     update_value + unit_feature_value.embed_w_index(),
                                     update_value + unit_feature_value.embed_g2sum_index(),
                                     push_value + DownpourDoubleUnitPushValue::embed_g_index(), push_show);
        _embedx_sgd_rule->update_value(
                                      update_value + unit_feature_value.embedx_w_index(),
                                      update_value + unit_feature_value.embedx_g2sum_index(),
                                      push_value + DownpourDoubleUnitPushValue::embedx_g_index(), push_show);
    }
    return 0;
}
bool DownpourDoubleUnitAccessor::create_value(int stage, const float* value) {
    // stage == 0, pull
    // stage == 1, push
    if (stage == 0) {
        return true;
    } else if (stage == 1) {
        auto show = DownpourDoubleUnitPushValue::show(const_cast<float*>(value));
        auto click = DownpourDoubleUnitPushValue::click(const_cast<float*>(value));
        auto score = show_click_score(show, click);
        if (score <= 0) {
            return false;
        }
        if (score >= 1) {
            return true;
        }
        return local_uniform_real_distribution<float>()(local_random_engine()) < score;
    } else {
        return true;
    }
}
double DownpourDoubleUnitAccessor::show_click_score(double show, double click) {
    //auto nonclk_coeff = _config.downpour_accessor_param().nonclk_coeff();
    //auto click_coeff = _config.downpour_accessor_param().click_coeff();
    auto nonclk_coeff = _config.downpour_accessor_param().nonclk_coeff();
    auto click_coeff = _config.downpour_accessor_param().click_coeff();
    return (show - click) * nonclk_coeff + click * click_coeff;
}
std::string DownpourDoubleUnitAccessor::parse_to_string(const float* v, int param_size) {
    thread_local std::ostringstream os;
    os.clear();
    os.str("");
    os << v[0] << " "
       << v[1] << " "
       << (float)((double*)(v + 2))[0] << " "
       << (float)((double*)(v + 4))[0] << " "
       << v[6] << " "
       << v[7];
    
    for (int i = unit_feature_value.embed_g2sum_index(); 
        i < unit_feature_value.embedx_w_index(); 
        i++) {
        os << " " << v[i];
    }
    auto show = unit_feature_value.show(const_cast<float*>(v));
    auto click = unit_feature_value.click(const_cast<float*>(v));
    auto score = show_click_score(show, click);
    if (score >= _config.embedx_threshold() && param_size > 9) {
        for (auto i = unit_feature_value.embedx_w_index(); 
            i < unit_feature_value.embedx_g2sum_index() + unit_feature_value.embedx_sgd_dim; 
            ++i) {
            os << " " << v[i];
        }
    }
    return os.str();
}
int DownpourDoubleUnitAccessor::parse_from_string(const std::string& str, float* value) {
    int embedx_dim = _config.embedx_dim();
    float data_buff[dim() + 2];
    float* data_buff_ptr = data_buff;
    _embedx_sgd_rule->init_value(
                                data_buff_ptr + unit_feature_value.embedx_w_index(),
                                data_buff_ptr + unit_feature_value.embedx_g2sum_index());
    auto str_len = str_to_float(str.data(), data_buff_ptr);
    CHECK(str_len >= 6) << "expect more than 6 real:" << str_len;
    int show_index = unit_feature_value.show_index();
    int click_index = unit_feature_value.click_index();
    int slot_index = unit_feature_value.slot_index();
    // no slot, embedx
    int value_dim = dim();
    int embedx_g2sum_index = unit_feature_value.embedx_g2sum_index();
    value[unit_feature_value.slot_index()] = -1;
    // copy unseen_days..delta_score
    memcpy(value, data_buff_ptr, show_index * sizeof(float));
    // copy show & click
    *(double*)(value + show_index) = (double)data_buff_ptr[2];
    *(double*)(value + click_index) = (double)data_buff_ptr[3];
    // copy embed_w..embedx_w
    memcpy(value + slot_index, data_buff_ptr + 4, (str_len - 4) * sizeof(float));
    return str_len + 2;
}

// for unit begin
int DownpourUnitAccessor::initialize() {
    auto name = _config.embed_sgd_param().name();
    _embed_sgd_rule = global_sparse_value_sgd_rule_factory().produce(name);
    _embed_sgd_rule->load_config(_config.embed_sgd_param(), 1);

    name = _config.embedx_sgd_param().name();
    _embedx_sgd_rule = global_sparse_value_sgd_rule_factory().produce(name);
    _embedx_sgd_rule->load_config(_config.embedx_sgd_param(), _config.embedx_dim());
   
    unit_feature_value.embed_sgd_dim = _embed_sgd_rule->dim();
    unit_feature_value.embedx_dim = _config.embedx_dim();
    unit_feature_value.embedx_sgd_dim = _embedx_sgd_rule->dim();
    _show_click_decay_rate = _config.downpour_accessor_param().show_click_decay_rate();
    
    return 0;
}

size_t DownpourUnitAccessor::dim() {
    return unit_feature_value.dim();
}

size_t DownpourUnitAccessor::dim_size(size_t dim) {
    auto embedx_dim = _config.embedx_dim();
    return unit_feature_value.dim_size(dim, embedx_dim);
}

size_t DownpourUnitAccessor::size() {
    return unit_feature_value.size();
}

size_t DownpourUnitAccessor::mf_size() {
    return (_config.embedx_dim() + unit_feature_value.embedx_sgd_dim) * sizeof(float);//embedx embedx_g2sum
}

// pull value
size_t DownpourUnitAccessor::select_dim() {
    auto embedx_dim = _config.embedx_dim();
    return 3 + embedx_dim;
}

size_t DownpourUnitAccessor::select_dim_size(size_t dim) {
    return sizeof(float);
}

size_t DownpourUnitAccessor::select_size() {
    return select_dim() * sizeof(float);
}

// push value
size_t DownpourUnitAccessor::update_dim() {
    auto embedx_dim = _config.embedx_dim();
    return 4 + embedx_dim;
}

size_t DownpourUnitAccessor::update_dim_size(size_t dim) {
    return sizeof(float);
}

size_t DownpourUnitAccessor::update_size() {
    return update_dim() * sizeof(float);
}

bool DownpourUnitAccessor::shrink(float* value) {
    //auto base_threshold = _config.downpour_accessor_param().base_threshold();
    //auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    //auto delete_threshold = _config.downpour_accessor_param().delete_threshold();
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delete_after_unseen_days = _config.downpour_accessor_param().delete_after_unseen_days();
    auto delete_threshold = _config.downpour_accessor_param().delete_threshold();

    // time_decay first
    unit_feature_value.show(value) *= _show_click_decay_rate;
    unit_feature_value.click(value) *= _show_click_decay_rate;

    // shrink after
    auto score = show_click_score(unit_feature_value.show(value), unit_feature_value.click(value));
    auto unseen_days = unit_feature_value.unseen_days(value);
    if (score < delete_threshold || unseen_days > delete_after_unseen_days) {
        return true;
    }
    return false;
}

bool DownpourUnitAccessor::save_ssd(float* value) {
    if (unit_feature_value.unseen_days(value) > _ssd_unseenday_threshold) {
        return true;
    }
    return false;
}

bool DownpourUnitAccessor::save_cache(
        float* value, int param, double global_cache_threshold) {
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    if (show_click_score(unit_feature_value.show(value), unit_feature_value.click(value)) >= base_threshold
        && unit_feature_value.unseen_days(value) <= delta_keep_days) {
        return unit_feature_value.show(value) > global_cache_threshold;
    }
    return false;
}

bool DownpourUnitAccessor::save(float* value, int param) {
    //auto base_threshold = _config.downpour_accessor_param().base_threshold();
    //auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    //auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    if (param == 2) {
        delta_threshold = 0;
    }
    switch (param) {
        // save all
        case 0:
            {
                return true;
            }
        // save xbox delta
        case 1:
        // save xbox base
        case 2:
            {
                if (show_click_score(unit_feature_value.show(value), unit_feature_value.click(value)) >= base_threshold
                        && unit_feature_value.delta_score(value) >= delta_threshold
                        && unit_feature_value.unseen_days(value) <= delta_keep_days) {
                    //do this after save, because it must not be modified when retry
                    if (param == 2) {
                        unit_feature_value.delta_score(value) = 0;
                    }
                    return true;
                } else {
                    return false;
                }
            }
        // already decayed in shrink 
        case 3:
            {
                //do this after save, because it must not be modified when retry
                //unit_feature_value.unseen_days(value)++;
                return true;
            }
        // save revert batch_model
        case 5:
            {
                return true;
            }
        default:
            return true;
    };
}

void DownpourUnitAccessor::update_stat_after_save(float* value, int param) {
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    if (param == 2) {
        delta_threshold = 0;
    }
    switch (param) {
        case 1:
            {
                if (show_click_score(unit_feature_value.show(value), unit_feature_value.click(value)) >= base_threshold
                        && unit_feature_value.delta_score(value) >= delta_threshold
                        && unit_feature_value.unseen_days(value) <= delta_keep_days) {
                    unit_feature_value.delta_score(value) = 0;
                }
            }
            return;
         case 3:
            {
                unit_feature_value.unseen_days(value)++;
            }
            return;
         default:
            return;
    };
}

int32_t DownpourUnitAccessor::create(float** values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* value = values[value_item];
        value[unit_feature_value.unseen_days_index()] = 0;
        value[unit_feature_value.delta_score_index()] = 0;
        value[unit_feature_value.show_index()] = 0;
        value[unit_feature_value.click_index()] = 0;
        value[unit_feature_value.slot_index()] = -1;
        _embed_sgd_rule->init_value(
            value + unit_feature_value.embed_w_index(),
            value + unit_feature_value.embed_g2sum_index());
        _embedx_sgd_rule->init_value(
            value + unit_feature_value.embedx_w_index(),
            value + unit_feature_value.embedx_g2sum_index());
    }
    return 0;
}

bool DownpourUnitAccessor::need_extend_mf(float* value) {
    float show = value[unit_feature_value.show_index()];
    float click = value[unit_feature_value.click_index()];
    //float score = (show - click) * _config.downpour_accessor_param().nonclk_coeff()
    float score = (show - click) * _config.downpour_accessor_param().nonclk_coeff()
        + click * _config.downpour_accessor_param().click_coeff();
        //+ click * _config.downpour_accessor_param().click_coeff();
    return score >= _config.embedx_threshold();
}

bool DownpourUnitAccessor::has_mf(size_t size) {
    return size > unit_feature_value.embedx_g2sum_index();
}

// from UnitFeatureValue to DownpourUnitPullValue
int32_t DownpourUnitAccessor::select(float** select_values, const float** values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* select_value = select_values[value_item];
        float* value = const_cast<float*>(values[value_item]);
        select_value[DownpourUnitPullValue::show_index()] = value[unit_feature_value.show_index()];
        select_value[DownpourUnitPullValue::click_index()] = value[unit_feature_value.click_index()];
        select_value[DownpourUnitPullValue::embed_w_index()] = value[unit_feature_value.embed_w_index()];
        memcpy(select_value + DownpourUnitPullValue::embedx_w_index(), 
            value + unit_feature_value.embedx_w_index(), embedx_dim * sizeof(float));
    }
    return 0;
}

// from DownpourUnitPushValue to DownpourUnitPushValue
// first dim: item
// second dim: field num
int32_t DownpourUnitAccessor::merge(float** update_values, const float** other_update_values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    size_t total_dim = DownpourUnitPushValue::dim(embedx_dim);
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* update_value = update_values[value_item];
        const float* other_update_value = other_update_values[value_item];
        for (auto i = 0u; i < total_dim; ++i) {
            if (i != DownpourUnitPushValue::slot_index()) {
                update_value[i] += other_update_value[i];
            }
        }
    }
    return 0;
}

// from DownpourUnitPushValue to UnitFeatureValue
// first dim: item
// second dim: field num
int32_t DownpourUnitAccessor::update(float** update_values, const float** push_values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* update_value = update_values[value_item];
        const float* push_value = push_values[value_item];
        float push_show = push_value[DownpourUnitPushValue::show_index()];
        float push_click = push_value[DownpourUnitPushValue::click_index()];
        float slot = push_value[DownpourUnitPushValue::slot_index()];
        update_value[unit_feature_value.show_index()] += push_show;
        update_value[unit_feature_value.click_index()] += push_click;
        update_value[unit_feature_value.slot_index()] = slot;
        update_value[unit_feature_value.delta_score_index()] +=
            (push_show - push_click) * _config.downpour_accessor_param().nonclk_coeff() +
            push_click * _config.downpour_accessor_param().click_coeff();
            //(push_show - push_click) * _config.downpour_accessor_param().nonclk_coeff() +
            //push_click * _config.downpour_accessor_param().click_coeff();
        update_value[unit_feature_value.unseen_days_index()] = 0;
        _embed_sgd_rule->update_value(
            update_value + unit_feature_value.embed_w_index(),
            update_value + unit_feature_value.embed_g2sum_index(),
            push_value + DownpourUnitPushValue::embed_g_index());
        _embedx_sgd_rule->update_value(
            update_value + unit_feature_value.embedx_w_index(),
            update_value + unit_feature_value.embedx_g2sum_index(),
            push_value + DownpourUnitPushValue::embedx_g_index());
    }
    return 0;
}

bool DownpourUnitAccessor::create_value(int stage, const float* value) {
    // stage == 0, pull
    // stage == 1, push
    if (stage == 0) {
        return true;
    } else if (stage == 1) {
        auto show = DownpourUnitPushValue::show(const_cast<float*>(value));
        auto click = DownpourUnitPushValue::click(const_cast<float*>(value));
        auto score = show_click_score(show, click);
        if (score <= 0) {
            return false;
        }
        if (score >= 1) {
            return true;
        }
        return local_uniform_real_distribution<float>()(local_random_engine()) < score;
    } else {
        return true;
    }
}

float DownpourUnitAccessor::show_click_score(float show, float click) {
    //auto nonclk_coeff = _config.downpour_accessor_param().nonclk_coeff();
    //auto click_coeff = _config.downpour_accessor_param().click_coeff();
    auto nonclk_coeff = _config.downpour_accessor_param().nonclk_coeff();
    auto click_coeff = _config.downpour_accessor_param().click_coeff();
    return (show - click) * nonclk_coeff + click * click_coeff;
}

std::string DownpourUnitAccessor::parse_to_string(const float* v, int param) {
    thread_local std::ostringstream os;
    os.clear();
    os.str("");
    os << v[0] << " "
        << v[1] << " "
        << v[2] << " "
        << v[3] << " "
        << v[4] << " "
        << v[5];
    for (int i = unit_feature_value.embed_g2sum_index(); 
        i < unit_feature_value.embedx_w_index(); 
        i++) {
        os << " " << v[i];
    }
    auto show = unit_feature_value.show(const_cast<float*>(v));
    auto click = unit_feature_value.click(const_cast<float*>(v));
    auto score = show_click_score(show, click);
    if (score >= _config.embedx_threshold()) {
        for (auto i = unit_feature_value.embedx_w_index(); 
            i < unit_feature_value.dim(); 
            ++i) {
            os << " " << v[i];
        }
    }
    return os.str();
}

int DownpourUnitAccessor::parse_from_string(const std::string& str, float* value) {
    int embedx_dim = _config.embedx_dim();
    
    _embedx_sgd_rule->init_value( 
        value + unit_feature_value.embedx_w_index(), 
        value + unit_feature_value.embedx_g2sum_index());
    auto ret = str_to_float(str.data(), value);
    CHECK(ret >= 6) << "expect more than 6 real:" << ret;
    return ret;
}
// for unit end

// for common begin
int DownpourCommonAccessor::initialize() {
    auto name = _config.embed_sgd_param().name();
    //TODO
    _embed_sgd_rule = global_sparse_value_sgd_rule_factory().produce(name);
    _embed_sgd_rule->load_config(_config.embed_sgd_param(), 1);

    name = _config.embedx_sgd_param().name();
    _embedx_sgd_rule = global_sparse_value_sgd_rule_factory().produce(name);
    _embedx_sgd_rule->load_config(_config.embedx_sgd_param(), _config.embedx_dim());
   
    common_feature_value.embed_sgd_dim = _embed_sgd_rule->dim();
    common_feature_value.embedx_dim = _config.embedx_dim();
    common_feature_value.embedx_sgd_dim = _embedx_sgd_rule->dim();
    _show_click_decay_rate = _config.downpour_accessor_param().show_click_decay_rate();
    
    return 0;
}

size_t DownpourCommonAccessor::dim() {
    return common_feature_value.dim();
}

size_t DownpourCommonAccessor::dim_size(size_t dim) {
    auto embedx_dim = _config.embedx_dim();
    return common_feature_value.dim_size(dim, embedx_dim);
}

size_t DownpourCommonAccessor::size() {
    return common_feature_value.size();
}

size_t DownpourCommonAccessor::mf_size() {
    return (_config.embedx_dim() + common_feature_value.embedx_sgd_dim) * sizeof(float);//embedx embedx_g2sum
}

// pull value
size_t DownpourCommonAccessor::select_dim() {
    auto embedx_dim = _config.embedx_dim();
    return 1 + embedx_dim;
}

size_t DownpourCommonAccessor::select_dim_size(size_t dim) {
    return sizeof(float);
}

size_t DownpourCommonAccessor::select_size() {
    return select_dim() * sizeof(float);
}

// push value
size_t DownpourCommonAccessor::update_dim() {
    auto embedx_dim = _config.embedx_dim();
    return 4 + embedx_dim;
}

size_t DownpourCommonAccessor::update_dim_size(size_t dim) {
    return sizeof(float);
}

size_t DownpourCommonAccessor::update_size() {
    return update_dim() * sizeof(float);
}

bool DownpourCommonAccessor::shrink(float* value) {
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delete_after_unseen_days = _config.downpour_accessor_param().delete_after_unseen_days();
    auto delete_threshold = _config.downpour_accessor_param().delete_threshold();

    // time_decay first
    common_feature_value.show(value) *= _show_click_decay_rate;
    common_feature_value.click(value) *= _show_click_decay_rate;

    // shrink after
    auto score = show_click_score(common_feature_value.show(value), common_feature_value.click(value));
    auto unseen_days = common_feature_value.unseen_days(value);
    if (score < delete_threshold || unseen_days > delete_after_unseen_days) {
        return true;
    }
    return false;
}

bool DownpourCommonAccessor::save(float* value, int param) {
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    if (param == 2) {
        delta_threshold = 0;
    }
    switch (param) {
        // save all
        case 0:
            {
                return true;
            }
        // save xbox delta
        case 1:
        // save xbox base
        case 2:
            {
                if (show_click_score(common_feature_value.show(value), common_feature_value.click(value)) >= base_threshold
                        && common_feature_value.delta_score(value) >= delta_threshold
                        && common_feature_value.unseen_days(value) <= delta_keep_days) {
                    //do this after save, because it must not be modified when retry
                    if (param == 2) {
                        common_feature_value.delta_score(value) = 0;
                    }
                    return true;
                } else {
                    return false;
                }
            }
        // already decayed in shrink 
        case 3:
            {
                //do this after save, because it must not be modified when retry
                //common_feature_value.unseen_days(value)++;
                return true;
            }
        // save revert batch_model
        case 5:
            {
                return true;
            }
        default:
            return true;
    };
}

void DownpourCommonAccessor::update_stat_after_save(float* value, int param) {
    auto base_threshold = _config.downpour_accessor_param().base_threshold();
    auto delta_threshold = _config.downpour_accessor_param().delta_threshold();
    auto delta_keep_days = _config.downpour_accessor_param().delta_keep_days();
    if (param == 2) {
        delta_threshold = 0;
    }
    switch (param) {
        case 1:
            {
                if (show_click_score(common_feature_value.show(value), common_feature_value.click(value)) >= base_threshold
                        && common_feature_value.delta_score(value) >= delta_threshold
                        && common_feature_value.unseen_days(value) <= delta_keep_days) {
                    common_feature_value.delta_score(value) = 0;
                }
            }
            return;
         case 3:
            {
                common_feature_value.unseen_days(value)++;
            }
            return;
         default:
            return;
    };
}

int32_t DownpourCommonAccessor::create(float** values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* value = values[value_item];
        value[common_feature_value.unseen_days_index()] = 0;
        value[common_feature_value.delta_score_index()] = 0;
        value[common_feature_value.show_index()] = 0;
        value[common_feature_value.click_index()] = 0;
        value[common_feature_value.slot_index()] = -1;
        _embed_sgd_rule->init_value(
            value + common_feature_value.embed_w_index(),
            value + common_feature_value.embed_g2sum_index());
        _embedx_sgd_rule->init_value(
            value + common_feature_value.embedx_w_index(),
            value + common_feature_value.embedx_g2sum_index());
    }
    return 0;
}

bool DownpourCommonAccessor::need_extend_mf(float* value) {
    float show = value[common_feature_value.show_index()];
    float click = value[common_feature_value.click_index()];
    //float score = (show - click) * _config.downpour_accessor_param().nonclk_coeff()
    float score = (show - click) * _config.downpour_accessor_param().nonclk_coeff()
        + click * _config.downpour_accessor_param().click_coeff();
        //+ click * _config.downpour_accessor_param().click_coeff();
    return score >= _config.embedx_threshold();
}

bool DownpourCommonAccessor::has_mf(size_t size) {
    return size > common_feature_value.embedx_g2sum_index();
}

// from CommonFeatureValue to DownpourCommonPullValue
int32_t DownpourCommonAccessor::select(float** select_values, const float** values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* select_value = select_values[value_item];
        float* value = const_cast<float*>(values[value_item]);
        //select_value[DownpourCommonPullValue::show_index()] = value[common_feature_value.show_index()];
        //select_value[DownpourCommonPullValue::click_index()] = value[common_feature_value.click_index()];
        select_value[DownpourCommonPullValue::embed_w_index()] = value[common_feature_value.embed_w_index()];
        memcpy(select_value + DownpourCommonPullValue::embedx_w_index(), 
            value + common_feature_value.embedx_w_index(), embedx_dim * sizeof(float));
    }
    return 0;
}

// from DownpourCommonPushValue to DownpourCommonPushValue
// first dim: item
// second dim: field num
int32_t DownpourCommonAccessor::merge(float** update_values, const float** other_update_values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    size_t total_dim = DownpourCommonPushValue::dim(embedx_dim);
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* update_value = update_values[value_item];
        const float* other_update_value = other_update_values[value_item];
        for (auto i = 0u; i < total_dim; ++i) {
            if (i != DownpourCommonPushValue::slot_index()) {
                update_value[i] += other_update_value[i];
            }
        }
    }
    return 0;
}

// from DownpourCommonPushValue to CommonFeatureValue
// first dim: item
// second dim: field num
int32_t DownpourCommonAccessor::update(float** update_values, const float** push_values, size_t num) {
    auto embedx_dim = _config.embedx_dim();
    for (size_t value_item = 0; value_item < num; ++value_item) {
        float* update_value = update_values[value_item];
        const float* push_value = push_values[value_item];
        float push_show = push_value[DownpourCommonPushValue::show_index()];
        float push_click = push_value[DownpourCommonPushValue::click_index()];
        float slot = push_value[DownpourCommonPushValue::slot_index()];
        update_value[common_feature_value.show_index()] += push_show;
        update_value[common_feature_value.click_index()] += push_click;
        update_value[common_feature_value.slot_index()] = slot;
        update_value[common_feature_value.delta_score_index()] +=
            (push_show - push_click) * _config.downpour_accessor_param().nonclk_coeff() +
            push_click * _config.downpour_accessor_param().click_coeff();
            //(push_show - push_click) * _config.downpour_accessor_param().nonclk_coeff() +
            //push_click * _config.downpour_accessor_param().click_coeff();
        update_value[common_feature_value.unseen_days_index()] = 0;
        _embed_sgd_rule->update_value(
            update_value + common_feature_value.embed_w_index(),
            update_value + common_feature_value.embed_g2sum_index(),
            push_value + DownpourCommonPushValue::embed_g_index());
        _embedx_sgd_rule->update_value(
            update_value + common_feature_value.embedx_w_index(),
            update_value + common_feature_value.embedx_g2sum_index(),
            push_value + DownpourCommonPushValue::embedx_g_index());
    }
    return 0;
}

bool DownpourCommonAccessor::create_value(int stage, const float* value) {
    // stage == 0, pull
    // stage == 1, push
    if (stage == 0) {
        return true;
    } else if (stage == 1) {
        auto show = DownpourCommonPushValue::show(const_cast<float*>(value));
        auto click = DownpourCommonPushValue::click(const_cast<float*>(value));
        auto score = show_click_score(show, click);
        if (score <= 0) {
            return false;
        }
        if (score >= 1) {
            return true;
        }
        //TODO: common
        return local_uniform_real_distribution<float>()(local_random_engine()) < score;
    } else {
        return true;
    }
}

float DownpourCommonAccessor::show_click_score(float show, float click) {
    //auto nonclk_coeff = _config.downpour_accessor_param().nonclk_coeff();
    //auto click_coeff = _config.downpour_accessor_param().click_coeff();
    auto nonclk_coeff = _config.downpour_accessor_param().nonclk_coeff();
    auto click_coeff = _config.downpour_accessor_param().click_coeff();
    return (show - click) * nonclk_coeff + click * click_coeff;
}

std::string DownpourCommonAccessor::parse_to_string(const float* v, int param) {
    thread_local std::ostringstream os;
    os.clear();
    os.str("");
    os << v[0] << " "
        << v[1] << " "
        << v[2] << " "
        << v[3] << " "
        << v[4] << " "
        << v[5];
    for (int i = common_feature_value.embed_g2sum_index(); 
        i < common_feature_value.embedx_w_index(); 
        i++) {
        os << " " << v[i];
    }
    auto show = common_feature_value.show(const_cast<float*>(v));
    auto click = common_feature_value.click(const_cast<float*>(v));
    auto score = show_click_score(show, click);
    if (score >= _config.embedx_threshold()) {
        for (auto i = common_feature_value.embedx_w_index(); 
            i < common_feature_value.dim(); 
            ++i) {
            os << " " << v[i];
        }
    }
    return os.str();
}

int DownpourCommonAccessor::parse_from_string(const std::string& str, float* value) {
    int embedx_dim = _config.embedx_dim();
    
    _embedx_sgd_rule->init_value( 
        value + common_feature_value.embedx_w_index(), 
        value + common_feature_value.embedx_g2sum_index());
    auto ret = str_to_float(str.data(), value);
    CHECK(ret >= 6) << "expect more than 6 real:" << ret;
    return ret;
}
// for common end

}
}
