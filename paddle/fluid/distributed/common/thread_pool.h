#pragma once
#include <condition_variable>
#include <forward_list>
#include <future>
#include <list>
#include <mutex>
#include <queue>
#include "thread_queue.h"

namespace paddle {
namespace distributed {

    template <class T> class ThreadPool {
    public:
        ThreadPool<T>(uint32_t thread_num) { Init(thread_num); }
        ~ThreadPool<T>() {
            _tasks.disable();

            for (std::thread &t : _threads) {
                t.join();
            }
        }

        void Init(size_t thread_num) {
            for (; thread_num; --thread_num) {
                _threads.emplace_front([this]() {
                    for (;;) {
                        std::packaged_task<T()> task = _tasks.pop();
                        if (task.valid()) {
                            task();
                        } else {
                            break;
                        }
                    }
                });
            }
        }

        void destroy(){
            delete this;
        }
        template <class Callable, class... Args>
        std::future<T> AddTask(Callable &&func, Args &&... args) {
            std::packaged_task<T()> task(std::bind(std::forward<Callable>(func),
                                                   std::forward<Args>(args)...));
            std::future<T> result = task.get_future();
            _tasks.push(std::move(task));
            return result;
        }

    private:
        typedef thread_queue<std::packaged_task<T()>, store_value> queue_t;
        queue_t _tasks;

        std::forward_list<std::thread> _threads;
    };
    typedef ThreadPool<void> WorkerPool;

} //distributed
} //paddle 
