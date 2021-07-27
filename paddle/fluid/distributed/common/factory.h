#ifndef BAIDU_BAIDU_PSLIB_INCLUDE_COMMON_FACTORY_H
#define BAIDU_BAIDU_PSLIB_INCLUDE_COMMON_FACTORY_H
#include <map>
#include <memory>
#include <glog/logging.h>

namespace paddle {
namespace ps {

template<class T>
class Factory {
public:
    typedef std::function<std::shared_ptr<T>()> producer_t;

    template<class TT>
    void add(const std::string& name) {
        add(name, []()->std::shared_ptr<TT> {
            return std::make_shared<TT>();
        });
    }
    void add(const std::string& name, producer_t producer) {
        CHECK(_items.insert({name, producer}).second) << "Factory item[" <<
            name << "] already exists";
    }
    template<class TT = T>
    std::shared_ptr<TT> produce(const std::string& name) {
        auto it = _items.find(name);
        CHECK(it != _items.end()) << "Factory item[" << name << "] not found";
        std::shared_ptr<T> obj = it->second();
        CHECK(obj) << "Factor item is empty: " << name;
        std::shared_ptr<TT> x = std::dynamic_pointer_cast<TT>(obj);
        CHECK(x) << "Factory item[" << name << "] can not cast from " << typeid(
                 *obj).name() << " to " << typeid(TT).name();
        return x;
    }
private:
    std::map<std::string, producer_t> _items;
};

}
}
#endif
