#pragma once
#include <functional>
#include <memory>
#include <type_traits>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace paddle {
namespace distributed {

template <class T> struct node {
    T value;
    node *next;

    node() : next(nullptr) {}
    void assign(const T &t) { value = t; }
    void assign(T &&t) { value = std::move(t); }
    void assign(T *t) { value = *t; }

    template <class... Args> void construct(Args &&... args) {
        value = T(std::forward<Args>(args)...);
    }
};

template <class T> struct ptr_node {
    std::unique_ptr<T> value;
    ptr_node *next;

    ptr_node() : value(nullptr), next(nullptr) {}

    void assign() { value.reset(new T); }
    void assign(T &&t) { value.reset(new T(std::move(t))); }
    void assign(const T &t) { value.reset(new T(t)); }
    void assign(T *t) { value.reset(t); }

    template <class... Args> void construct(Args &&... args) {
        value.reset(new T(std::forward<Args>(args)...));
    }
};

struct store_value {};
struct store_ptr {};

template <class T, class StorageType> struct node_traits {};

template <class T> struct node_traits<T, store_value> {
    typedef node<T> node_type;
    typedef T value_type;
};

template <class T> struct node_traits<T, store_ptr> {
    typedef ptr_node<T> node_type;
    typedef std::unique_ptr<T> value_type;
};

template <class T, class StorageType> class thread_queue {
public:
    typedef typename node_traits<T, StorageType>::value_type value_type;

    thread_queue() : _size(0), _in_use(true) { _head = _tail = new node_type; }
    ~thread_queue() {
        for (; _head != nullptr;) {
            node_type *tmp = _head;
            _head = _head->next;
            delete tmp;
        }
    }

    void push(T &&value) {
        node_type *new_node = new node_type;
        {
            std::lock_guard<std::mutex> lck(_tail_mtx);
            _tail->assign(std::move(value));
            _tail->next = new_node;
            _tail = new_node;
        }
        ++_size;
        _cv.notify_one();
    }

    void push(const T &&value) {
        node_type *new_node = new node_type;
        {
            std::lock_guard<std::mutex> lck(_tail_mtx);
            _tail->assign(value);
            _tail->next = new_node;
            _tail = new_node;
        }
        ++_size;
        _cv.notify_one();
    }

    void push(T *value) {
        node_type *new_node = new node_type;
        {
            std::lock_guard<std::mutex> lck(_tail_mtx);
            _tail->assign(value);
            _tail->next = new_node;
            _tail = new_node;
        }
        ++_size;
        _cv.notify_one();
    }

    template <class... Args> void emplace(Args &&... args) {
        node_type *new_node = new node_type;
        {
            std::lock_guard<std::mutex> lck(_tail_mtx);
            _tail->construct(std::forward<Args>(args)...);
            _tail->next = new_node;
            _tail = new_node;
        }
        ++_size;
        _cv.notify_one();
    }

    value_type pop() {
        std::lock_guard<std::mutex> head_lck(_head_mtx);
        {
            std::unique_lock<std::mutex> tail_lck(_tail_mtx);
            _cv.wait(tail_lck, [=] { return _in_use == false || _head != _tail; });
        }

        if (_in_use == false) {
            return value_type();
        }

        node_type *next = _head->next;
        value_type ret = std::move(_head->value);

        delete _head;
        _head = next;
        --_size;

        return ret;
    }

    bool empty() {
        std::lock_guard<std::mutex> head_lck(_head_mtx);
        std::lock_guard<std::mutex> tail_lck(_tail_mtx);
        return _head == _tail;
    }
    int size() {
        return _size;
    }

    void enable() {
        _in_use = true;
        _cv.notify_all();
    }
    void disable() {
        _in_use = false;
        _cv.notify_all();
    }

private:
    typedef typename node_traits<T, StorageType>::node_type node_type;

    node_type *get_tail() {
        std::lock_guard<std::mutex> lck(_tail_mtx);
        return _tail;
    }

    node_type *_head;
    node_type *_tail;

    std::atomic_int _size;
    std::atomic_bool _in_use;

    std::mutex _head_mtx;
    std::mutex _tail_mtx;
    std::condition_variable _cv;
};

} // pslib
} // paddle 
