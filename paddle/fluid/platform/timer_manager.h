#pragma once
#include <stdlib.h>
#include <vector>
#include <cstddef>
#include <stdint.h>
#include "gflags/gflags.h"
#include <string>
DECLARE_bool(debug_timermanager);
namespace paddle {
namespace platform {
#define TIMER_SINGLETHREAD_ENTER(timer_id) \
    if (FLAGS_debug_timermanager) {\
        platform::TimerManager* timer_manager = platform::TimerManager::instance();\
        timer_manager->enter(timer_id);\
    }

#define TIMER_SINGLETHREAD_LEAVE(timer_id) \
    if (FLAGS_debug_timermanager) {\
        platform::TimerManager* timer_manager = platform::TimerManager::instance();\
        timer_manager->leave(timer_id);\
    }
/*
#define TIMER_SINGLETHREAD_STATISTICS(timer_id) \
    if (FLAGS_debug_timermanager) {\
        platform::TimerManager* timer_manager = platform::TimerManager::instance();\
        uint64_t avg = 0;\
        uint64_t cnt = timer_manager->count(timer_id);\
        if (cnt != 0) {\
            avg = timer_manager->elapse(timer_id) / cnt;\
        }\
        VLOG(0) << #timer_id << " avg|total|cnt=(" << timer_manager->get_format_str(avg) << "|" << timer_manager->get_format_str(timer_manager->elapse(timer_id))\
                << "|" << timer_manager->count(timer_id) << ")";\
    }
*/
#define TIMER_SINGLETHREAD_STATISTICS(timer_id) \
    if (FLAGS_debug_timermanager) {\
        platform::TimerManager* timer_manager = platform::TimerManager::instance();\
        uint64_t avg = 0;\
        uint64_t cnt = timer_manager->count(timer_id);\
        if (cnt != 0) {\
            avg = timer_manager->elapse(timer_id) / cnt;\
        }\
        VLOG(0) << #timer_id << " avg|total|cnt=(" << avg << "|" << timer_manager->elapse(timer_id)\
                << "|" << timer_manager->count(timer_id) << ")";\
    }

#define TIMER_MULTITHREAD_ENTER(timer_id, thread_id) \
    if (FLAGS_debug_timermanager) {\
        platform::TimerManager* timer_manager = platform::TimerManager::instance();\
        timer_manager->enter(timer_id + thread_id * platform::TIMER_MAX_ID);\
    }

#define TIMER_MULTITHREAD_LEAVE(timer_id, thread_id) \
    if (FLAGS_debug_timermanager) {\
        platform::TimerManager* timer_manager = platform::TimerManager::instance();\
        timer_manager->leave(timer_id + thread_id * platform::TIMER_MAX_ID);\
    }

#define TIMER_MULTITHREAD_STATISTICS(timer_id, thread_id) \
    if (FLAGS_debug_timermanager) {\
        platform::TimerManager* timer_manager = platform::TimerManager::instance();\
        VLOG(0) << "thread_id: "<< thread_id<< " "<< #timer_id << " total time "<< timer_manager->elapse(timer_id + thread_id * platform::TIMER_MAX_ID)\
                << " count" << timer_manager->count(timer_id + thread_id * platform::TIMER_MAX_ID) << " avg " << timer_manager->elapse(timer_id + thread_id * platform::TIMER_MAX_ID) * 1.0 / timer_manager->count(timer_id + thread_id * platform::TIMER_MAX_ID);\
    }
/* 
#define TIMER_MULTITHREAD_STATISTICS_ALL(timer_id, thread_num) \
    if (FLAGS_debug_timermanager) {\
        platform::TimerManager* timer_manager = platform::TimerManager::instance();\
        std::string debug_str = std::string(#timer_id) + " tid|avg|total|cnt=(";\
        std::string tid_str = "";\
        std::string avg_str = "";\
        std::string total_str = "";\
        std::string cnt_str = "";\
        for (int tid = 0; tid < int(thread_num); ++tid) {\
            tid_str += std::to_string(tid);\
            uint64_t total = timer_manager->elapse(timer_id + tid * platform::TIMER_MAX_ID);\
            total_str += timer_manager->get_format_str(total);\
            uint64_t cnt = timer_manager->count(timer_id + tid * platform::TIMER_MAX_ID);\
            cnt_str += std::to_string(cnt);\
            uint64_t avg = 0;\
            if (cnt != 0) {\
                avg = total / cnt;\
            }\
            avg_str += timer_manager->get_format_str(avg);\
            if (tid != int(thread_num) - 1) {\
                tid_str += ",";\
                avg_str += ",";\
                total_str += ",";\
                cnt_str += ",";\
            }\
        }\
        debug_str += tid_str + "|" + avg_str + "|" + total_str + "|" + cnt_str + ")";\
        VLOG(0) << debug_str;\
    }
*/
#define TIMER_MULTITHREAD_STATISTICS_ALL(timer_id, thread_num) \
    if (FLAGS_debug_timermanager) {\
        platform::TimerManager* timer_manager = platform::TimerManager::instance();\
        std::string debug_str = std::string(#timer_id) + " tid|avg|total|cnt=(";\
        std::string tid_str = "";\
        std::string avg_str = "";\
        std::string total_str = "";\
        std::string cnt_str = "";\
        for (int tid = 0; tid < int(thread_num); ++tid) {\
            tid_str += std::to_string(tid);\
            uint64_t total = timer_manager->elapse(timer_id + tid * platform::TIMER_MAX_ID);\
            total_str += std::to_string(total);\
            uint64_t cnt = timer_manager->count(timer_id + tid * platform::TIMER_MAX_ID);\
            cnt_str += std::to_string(cnt);\
            uint64_t avg = 0;\
            if (cnt != 0) {\
                avg = total / cnt;\
            }\
            avg_str += std::to_string(avg);\
            if (tid != int(thread_num) - 1) {\
                tid_str += ",";\
                avg_str += ",";\
                total_str += ",";\
                cnt_str += ",";\
            }\
        }\
        debug_str += tid_str + "|" + avg_str + "|" + total_str + "|" + cnt_str + ")";\
        VLOG(0) << debug_str;\
    }


enum timer_ids{
    TIMER_TRAINER_ALL = 1,
    TIMER_TRAINER_INIT = 2,
    TIMER_TRAINER_CREATE_WORKER = 3,
    TIMER_TRAINER_WORKERS = 4,
    TIMER_TRAINER_FINALIZE = 5,
    TIMER_WORKER_ALL = 6,
    TIMER_WORKER_EACH_BATCH = 7,
    TIMER_WORKER_FEED_NEXT = 8,
    TIMER_WORKER_PULL_BOX_SPARSE = 9,
    TIMER_WORKER_PUSH_BOX_SPARSE = 10,
    TIMER_WORKER_ALL_OPS = 11,
    TIMER_WORKER_SKIP_OPS = 12,
    TIMER_OPS_PUSH_SPARSE_GRAD_ALL = 13,
    TIMER_OPS_PUSH_SPARSE_GRAD_DO = 14,
    TIMER_OPS_PUSH_SPARSE_GRAD_COPY = 15,
    TIMER_OPS_PUSH_SPARSE_GRAD_PUSH = 16,
    TIMER_OPS_PUSH_SPARSE_HETER_ALL = 17,
    TIMER_OPS_PUSH_SPARSE_HETER_STAGE1 = 18,
    TIMER_OPS_PUSH_SPARSE_HETER_STAGE2 = 19,
    TIMER_OPS_PUSH_SPARSE_HETER_STAGE3 = 20,
    TIMER_OPS_PUSH_SPARSE_HETER_STAGE4 = 21,
    TIMER_OPS_PUSH_SPARSE_HETER_STAGE5 = 22,
    TIMER_OPS_PULL_SPARSE_ALL = 23,
    TIMER_OPS_PULL_SPARSE_DO = 24,
    TIMER_OPS_PULL_SPARSE_DO_FIRST = 25,
    TIMER_OPS_PULL_SPARSE_FID2BFID = 26,
    TIMER_OPS_PULL_SPARSE_PULL = 27,
    TIMER_OPS_PULL_SPARSE_COPY_FOR_PULL = 28,
    TIMER_OPS_PULL_SPARSE_HETER_ALL = 29,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE1 = 30,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE2 = 31,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE3 = 32,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE4 = 33,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE5 = 34,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE6 = 35,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE7 = 36,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE8 = 37,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE9 = 38,
    TIMER_OPS_PULL_SPARSE_HETER_STAGE10 = 39,
    TIMER_OPS_PULL_SPARSE_COPY = 40,
    TIMER_MAX_ID
};

class TimerManager {
public:
    static TimerManager* instance() {
        static TimerManager _instance;
        return &_instance;
    }
    ~TimerManager();
    void init(uint32_t max_state_id);
    void enter(uint32_t state);
    void leave(uint32_t state);
    uint64_t elapse(uint32_t state) const;
    uint64_t count(uint32_t state) const;
    std::string get_format_str(uint64_t num);
private:
    TimerManager() {};
    //explicit TimerManager(uint32_t max_state_id);
    struct state_t {
        uint64_t    last_enter_time;
        uint64_t    total_time;
        uint64_t    total_count;
    };
    bool is_already_enter(uint32_t state_id) const;
    uint64_t elapse_time(uint64_t start, uint64_t end) const;
    void reset_state(state_t& state) const;
    state_t& state(uint32_t state_id);
    const state_t& state(uint32_t state_id) const;
    uint64_t current_time() const;
private:
    state_t _empty;
    std::vector<state_t>    _states;
};

class ScopedStateTimer {
public:
    ScopedStateTimer(TimerManager* timer, uint32_t state) :
        _p_timer(timer),
        _state(state)
    {
        _p_timer->enter(_state);
    }

    ~ScopedStateTimer()
    {
        _p_timer->leave(_state);
    }

private:
    TimerManager*     _p_timer;
    uint32_t        _state;
};

}  // namespace platform
}  // namespace paddle
