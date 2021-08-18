#ifndef MAPBOX_UTIL_VARIANT_VISITOR_HPP
#define MAPBOX_UTIL_VARIANT_VISITOR_HPP

#include <utility>

namespace paddle {

template <typename... Fns>
struct visitor;

template <typename Fn>
struct visitor<Fn> : Fn {
  using Fn::operator();

  template <typename T>
  visitor(T&& fn) : Fn(std::forward<T>(fn)) {}
};

template <typename Fn, typename... Fns>
struct visitor<Fn, Fns...> : Fn, visitor<Fns...> {
  using Fn::operator();
  using visitor<Fns...>::operator();

  template <typename T, typename... Ts>
  visitor(T&& fn, Ts&&... fns)
      : Fn(std::forward<T>(fn)), visitor<Fns...>(std::forward<Ts>(fns)...) {}
};

template <typename... Fns>
visitor<typename std::decay<Fns>::type...> make_visitor(Fns&&... fns) {
  return visitor<typename std::decay<Fns>::type...>(std::forward<Fns>(fns)...);
}

}  // namespace paddle

#endif  // MAPBOX_UTIL_VARIANT_VISITOR_HPP
