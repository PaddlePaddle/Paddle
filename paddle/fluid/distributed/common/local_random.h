#ifndef BAIDU_BAIDU_PSLIB_INCLUDE_COMMON_LOCAL_RANDOM_H
#define BAIDU_BAIDU_PSLIB_INCLUDE_COMMON_LOCAL_RANDOM_H
#include <random>
#include <atomic>
#include <time.h>
#include <assert.h>

namespace paddle {
namespace ps {

// Get time in seconds.
inline double current_realtime() {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return tp.tv_sec + tp.tv_nsec * 1e-9;
}

inline std::default_random_engine& local_random_engine() {
    struct engine_wrapper_t {
        std::default_random_engine engine;
        engine_wrapper_t() {
            static std::atomic<unsigned long> x(0);
            std::seed_seq sseq = {x++, x++, x++, (unsigned long)(current_realtime() * 1000)};
            engine.seed(sseq);
        }
    };
    thread_local engine_wrapper_t r;
    return r.engine;
}

template<class T = double>
std::uniform_real_distribution<T>& local_uniform_real_distribution() {
    thread_local std::uniform_real_distribution<T> distr;
    assert(distr.a() == 0.0 && distr.b() == 1.0);
    return distr;
}

template<class T = double>
T uniform_real() {
    return local_uniform_real_distribution<T>()(local_random_engine());
}

template<class T = double>
T uniform_real(T a, T b) {
    if (a == b) {
        return a;
    }
    return (T)(a + uniform_real<T>() * (b - a));
}
}
}
#endif
