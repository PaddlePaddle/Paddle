#include "paddle/fluid/platform/timer_manager.h"
#include <sys/time.h>

namespace paddle {
namespace platform {

TimerManager::~TimerManager() {
    // do nothing.
}

void TimerManager::init(uint32_t max_state_id) {
    reset_state(_empty);
    _states.resize(max_state_id + 1);
    for (size_t i = 0; i < _states.size(); ++i) {
        reset_state(_states[i]);
    }
}

void TimerManager::enter(uint32_t state_id) {
    if (is_already_enter(state_id)) {
        return;
    }
    uint64_t curr_time = current_time();
    state_t& st = state(state_id);
    st.last_enter_time = curr_time;
    ++st.total_count;
}

void TimerManager::leave(uint32_t state_id) {
    if (!is_already_enter(state_id)) {
        return;
    }
    uint64_t curr_time = current_time();
    state_t& st = state(state_id);
    st.total_time += elapse_time(st.last_enter_time, curr_time);
    st.last_enter_time = 0;
}

uint64_t TimerManager::elapse(uint32_t state_id) const {
    const state_t& st = state(state_id);
    return st.total_time;
}

uint64_t TimerManager::count(uint32_t state_id) const {
    const state_t& st = state(state_id);
    return st.total_count;
}

bool TimerManager::is_already_enter(uint32_t state_id) const {
    const state_t& st = state(state_id);
    return (st.last_enter_time != 0);
}

void TimerManager::reset_state(state_t& st) const {
    st.last_enter_time = 0;
    st.total_time = 0;
    st.total_count = 0;
}

const TimerManager::state_t& TimerManager::state(uint32_t state_id) const {
    return const_cast<TimerManager*>(this)->state(state_id);
}

TimerManager::state_t& TimerManager::state(uint32_t state_id) {
    if (state_id >= _states.size()) {
        return _empty;
    }
    return _states[state_id];
}

uint64_t TimerManager::current_time() const {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000UL + tv.tv_usec;
}

std::string TimerManager::get_format_str(uint64_t num) {
    std::string res = "";
    int S = num / 1000000;
    int MS = (num - S * 1000000) / 1000;
    int US = num - S * 1000000 - MS * 1000;
    if (num >= 1000000) {
        res += std::to_string(S) + "s";
    }
    if (num >= 1000) {
        res += std::to_string(MS) + "ms";
    }
    res += std::to_string(US) + "us";
    return res;
}

uint64_t TimerManager::elapse_time(uint64_t start, uint64_t end) const {
    if (start > end) {
        return 0;
    }
    return (end - start);
}

}  // namespace platform
}  // namespace paddle
