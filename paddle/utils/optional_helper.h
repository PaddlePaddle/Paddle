#ifndef PADDLE_OPTIONAL_HELPER
#define PADDLE_OPTIONAL_HELPER

#include <type_traits>

namespace paddle {

#ifdef __GNUC__
__extension__ typedef long long long_long_type;
__extension__ typedef unsigned long long ulong_long_type;
#else
typedef long long long_long_type;
typedef unsigned long long ulong_long_type;
#endif

namespace detail {

#ifdef PADDLE_MSVC
#pragma warning(push)
#pragma warning( \
    disable : 4324)  // structure was padded due to __declspec(align())
#endif
template <typename T>
struct alignment_of_hack {
  char c;
  T t;
  alignment_of_hack();
};
#ifdef PADDLE_MSVC
#pragma warning(pop)
#endif

template <unsigned A, unsigned S>
struct alignment_logic {
  static const std::size_t value = A < S ? A : S;
};

template <typename T>
struct alignment_of_impl {
#if defined(PADDLE_MSVC) && (PADDLE_MSVC >= 1400)
  //
  // With MSVC both the native __alignof operator
  // and our own logic gets things wrong from time to time :-(
  // Using a combination of the two seems to make the most of a bad job:
  //
  static const std::size_t value =
      (::paddle::detail::alignment_logic<
          sizeof(::paddle::detail::alignment_of_hack<T>) - sizeof(T),
          __alignof(T)>::value);
#elif !defined(PADDLE_ALIGNMENT_OF)
  static const std::size_t value =
      (::paddle::detail::alignment_logic<
          sizeof(::paddle::detail::alignment_of_hack<T>) - sizeof(T),
          sizeof(T)>::value);
#else
  //
  // We put this here, rather than in the definition of
  // alignment_of below, because MSVC's __alignof doesn't
  // always work in that context for some unexplained reason.
  // (See type_with_alignment tests for test cases).
  //
  static const std::size_t value = PADDLE_ALIGNMENT_OF(T);
#endif
};

}  // namespace detail

template <typename T>
struct alignment_of
    : ::std::integral_constant<std::size_t,
                               ::paddle::detail::alignment_of_impl<T>::value> {
};

namespace mpl {

struct integral_c_tag {
  static const int value = 0;
};

template <typename T, T value>
struct integral_c;

// 'bool' constant doesn't have 'next'/'prior' members
template <bool C>
struct integral_c<bool, C> {
  static const bool value = C;
  typedef integral_c_tag tag;
  typedef integral_c type;
  typedef bool value_type;
  operator bool() const { return this->value; }
};

//  [JDG Feb-4-2003] made void_ a complete type to allow it to be
//  instantiated so that it can be passed in as an object that can be
//  used to select an overloaded function. Possible use includes signaling
//  a zero arity functor evaluation call.
struct void_ {
  typedef void_ type;
};

template <bool C_>
struct bool_ {
  static const bool value = C_;
  typedef integral_c_tag tag;
  typedef bool_ type;
  typedef bool value_type;
  operator bool() const { return this->value; }
};

// shorcuts
typedef bool_<true> true_;
typedef bool_<false> false_;

template <bool C, typename T1, typename T2>
struct if_c {
  typedef T1 type;
};

template <typename T1, typename T2>
struct if_c<false, T1, T2> {
  typedef T2 type;
};

// n.a. == not available
struct na {
  typedef na type;
  enum { value = 0 };
};

#define PADDLE_MPL_AUX_NA_PARAM(param) param = na

// agurt, 05/sep/04: nondescriptive parameter names for the sake of DigitalMars
// (and possibly MWCW < 8.0); see
// http://article.gmane.org/gmane.comp.lib.boost.devel/108959
template <typename PADDLE_MPL_AUX_NA_PARAM(T1),
          typename PADDLE_MPL_AUX_NA_PARAM(T2),
          typename PADDLE_MPL_AUX_NA_PARAM(T3)>
struct if_ {
 private:
  // agurt, 02/jan/03: two-step 'type' definition for the sake of aCC
  typedef if_c<static_cast<bool>(T1::value), T2, T3> almost_type_;

 public:
  typedef typename almost_type_::type type;

  // PADDLE_MPL_AUX_LAMBDA_SUPPORT(3,if_,(T1,T2,T3))
};

}  // namespace mpl

namespace detail {

#define PADDLE_PP_CAT_I(a, b) a##b
#define PADDLE_PP_CAT(a, b) PADDLE_PP_CAT_I(a, b)

#define PADDLE_PP_INC_0 1
#define PADDLE_PP_INC_1 2
#define PADDLE_PP_INC_2 3
#define PADDLE_PP_INC_3 4
#define PADDLE_PP_INC_4 5
#define PADDLE_PP_INC_5 6
#define PADDLE_PP_INC_6 7
#define PADDLE_PP_INC_7 8
#define PADDLE_PP_INC_8 9
#define PADDLE_PP_INC_9 10
#define PADDLE_PP_INC_10 11
#define PADDLE_PP_INC_11 12

#define PADDLE_PP_INC_I(x) PADDLE_PP_INC_##x
#define PADDLE_PP_INC(x) PADDLE_PP_INC_I(x)

template <bool found, std::size_t target, class TestType>
struct lower_alignment_helper {
  typedef char type;
  enum { value = true };
};

template <std::size_t target, class TestType>
struct lower_alignment_helper<false, target, TestType> {
  enum { value = (alignment_of<TestType>::value == target) };
  typedef typename mpl::if_c<value, TestType, char>::type type;
};

#define PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, I, T)                          \
  typename lower_alignment_helper<PADDLE_PP_CAT(found, I), target, T>::type \
      PADDLE_PP_CAT(t, I);                                                  \
  enum {                                                                    \
    PADDLE_PP_CAT(found, PADDLE_PP_INC(I)) =                                \
        lower_alignment_helper<PADDLE_PP_CAT(found, I), target, T>::value   \
  };

#define PADDLE_TT_CHOOSE_T(R, P, I, T) T PADDLE_PP_CAT(t, I);

class alignment_dummy;
typedef void (*function_ptr)();
typedef int(alignment_dummy::*member_ptr);
typedef int (alignment_dummy::*member_function_ptr)();

template <std::size_t target>
union lower_alignment {
  enum { found0 = false };

  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 0, char);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 1, short);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 2, int);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 3, long);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 4, ::paddle::long_long_type);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 5, float);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 6, double);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 7, long double);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 8, void*);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 9, function_ptr);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 10, member_ptr);
  PADDLE_TT_CHOOSE_MIN_ALIGNMENT(R, P, 11, member_function_ptr);
};

union max_align {
  PADDLE_TT_CHOOSE_T(R, P, 0, char);
  PADDLE_TT_CHOOSE_T(R, P, 1, short);
  PADDLE_TT_CHOOSE_T(R, P, 2, int);
  PADDLE_TT_CHOOSE_T(R, P, 3, long);
  PADDLE_TT_CHOOSE_T(R, P, 4, ::paddle::long_long_type);
  PADDLE_TT_CHOOSE_T(R, P, 5, float);
  PADDLE_TT_CHOOSE_T(R, P, 6, double);
  PADDLE_TT_CHOOSE_T(R, P, 7, long double);
  PADDLE_TT_CHOOSE_T(R, P, 8, void*);
  PADDLE_TT_CHOOSE_T(R, P, 9, function_ptr);
  PADDLE_TT_CHOOSE_T(R, P, 10, member_ptr);
  PADDLE_TT_CHOOSE_T(R, P, 11, member_function_ptr);
};

// This alignment method originally due to Brian Parker, implemented by David
// Abrahams, and then ported here by Doug Gregor.
template <std::size_t TAlign, std::size_t Align>
struct is_aligned {
  static const bool value = (TAlign >= Align) & (TAlign % Align == 0);
};

template <std::size_t Align>
class type_with_alignment_imp {
  typedef ::paddle::detail::lower_alignment<Align> t1;
  typedef typename mpl::if_c<
      ::paddle::detail::is_aligned<::paddle::alignment_of<t1>::value,
                                   Align>::value,
      t1,
      ::paddle::detail::max_align>::type align_t;

  static const std::size_t found = alignment_of<align_t>::value;

  static_assert(found >= Align, "found >= Align");
  static_assert(found % Align == 0, "found % Align == 0");

 public:
  typedef align_t type;
};

template <std::size_t Align>
class type_with_alignment
    : public ::paddle::detail::type_with_alignment_imp<Align> {};

#if defined(__GNUC__)
namespace align {
struct __attribute__((__aligned__(2))) a2 {};
struct __attribute__((__aligned__(4))) a4 {};
struct __attribute__((__aligned__(8))) a8 {};
struct __attribute__((__aligned__(16))) a16 {};
struct __attribute__((__aligned__(32))) a32 {};
}

template <>
class type_with_alignment<1> {
 public:
  typedef char type;
};
template <>
class type_with_alignment<2> {
 public:
  typedef align::a2 type;
};
template <>
class type_with_alignment<4> {
 public:
  typedef align::a4 type;
};
template <>
class type_with_alignment<8> {
 public:
  typedef align::a8 type;
};
template <>
class type_with_alignment<16> {
 public:
  typedef align::a16 type;
};
template <>
class type_with_alignment<32> {
 public:
  typedef align::a32 type;
};

#endif

#if (defined(PADDLE_MSVC) || (defined(PADDLE_INTEL) && defined(_MSC_VER))) && \
    _MSC_VER >= 1300
//
// MSVC supports types which have alignments greater than the normal
// maximum: these are used for example in the types __m64 and __m128
// to provide types with alignment requirements which match the SSE
// registers.  Therefore we extend type_with_alignment<> to support
// such types, however, we have to be careful to use a builtin type
// whenever possible otherwise we break previously working code:
// see http://article.gmane.org/gmane.comp.lib.boost.devel/173011
// for an example and test case.  Thus types like a8 below will
// be used *only* if the existing implementation can't provide a type
// with suitable alignment.  This does mean however, that type_with_alignment<>
// may return a type which cannot be passed through a function call
// by value (and neither can any type containing such a type like
// Boost.Optional).  However, this only happens when we have no choice
// in the matter because no other "ordinary" type is available.
//
namespace align {
struct __declspec(align(8)) a8 {
  char m[8];
  typedef a8 type;
};
struct __declspec(align(16)) a16 {
  char m[16];
  typedef a16 type;
};
struct __declspec(align(32)) a32 {
  char m[32];
  typedef a32 type;
};
struct __declspec(align(64)) a64 {
  char m[64];
  typedef a64 type;
};
struct __declspec(align(128)) a128 {
  char m[128];
  typedef a128 type;
};
}

template <>
class type_with_alignment<8> {
  typedef mpl::if_c <
      ::paddle::alignment_of<detail::max_align>::
          value<8, align::a8, detail::type_with_alignment_imp<8>>::type t1;

 public:
  typedef t1::type type;
};
template <>
class type_with_alignment<16> {
  typedef mpl::if_c <
      ::paddle::alignment_of<detail::max_align>::
          value<16, align::a16, detail::type_with_alignment_imp<16>>::type t1;

 public:
  typedef t1::type type;
};
template <>
class type_with_alignment<32> {
  typedef mpl::if_c <
      ::paddle::alignment_of<detail::max_align>::
          value<32, align::a32, detail::type_with_alignment_imp<32>>::type t1;

 public:
  typedef t1::type type;
};
template <>
class type_with_alignment<64> {
  typedef mpl::if_c <
      ::paddle::alignment_of<detail::max_align>::
          value<64, align::a64, detail::type_with_alignment_imp<64>>::type t1;

 public:
  typedef t1::type type;
};
template <>
class type_with_alignment<128> {
  typedef mpl::if_c <
      ::paddle::alignment_of<detail::max_align>::
          value<128, align::a128, detail::type_with_alignment_imp<128>>::type
              t1;

 public:
  typedef t1::type type;
};

#endif

///////////////////////////////////////////////////////////////////////////////
// (detail) class template reference_content
//
// Non-Assignable wrapper for references.
//
template <typename RefT>
class reference_content {
 private:  // representation
  RefT content_;

 public:  // structors
  ~reference_content() {}

  reference_content(RefT r) : content_(r) {}

  reference_content(const reference_content& operand)
      : content_(operand.content_) {}

 private:  // non-Assignable
  reference_content& operator=(const reference_content&);

 public:  // queries
  RefT get() const { return content_; }
};

///////////////////////////////////////////////////////////////////////////////
// (detail) metafunction make_reference_content
//
// Wraps with reference_content if specified type is reference.
//

template <typename T = mpl::void_>
struct make_reference_content;

template <typename T>
struct make_reference_content {
  typedef T type;
};

template <typename T>
struct make_reference_content<T&> {
  typedef reference_content<T&> type;
};

template <>
struct make_reference_content<mpl::void_> {
  template <typename T>
  struct apply : make_reference_content<T> {};

  typedef mpl::void_ type;
};

}  // namespace detail
}

namespace paddle {
// alignment_of define
template <class T, T val>
struct integral_constant : public mpl::integral_c<T, val> {
  typedef integral_constant<T, val> type;
};

template <>
struct integral_constant<bool, true> : public mpl::true_ {
  typedef integral_constant<bool, true> type;
};

template <>
struct integral_constant<bool, false> : public mpl::false_ {
  typedef integral_constant<bool, false> type;
};

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

#ifdef _MSC_VER
#define PADDLE_MSVC _MSC_VER
#endif

#if !defined(unix) || defined(__LP64__)
// GCC sometimes lies about alignment requirements
// of type double on 32-bit unix platforms, use the
// old implementation instead in that case:
#define PADDLE_ALIGNMENT_OF(T) __alignof__(T)
#endif

template <typename T>
struct is_reference : ::paddle::integral_constant<bool, false> {};

template <typename T>
struct is_reference<T&> : ::paddle::integral_constant<bool, true> {};

// not_
namespace mpl {

namespace aux {

template <long C_>  // 'long' is intentional here
struct not_impl : bool_<!C_> {};

}  // namespace aux

template <typename T = na>
struct not_ : aux::not_impl<T::type::value> {};
}  // namespace mpl

// template<class OP> bool equal_pointees(OP const& x, OP const& y);
// template<class OP> struct equal_pointees_t;
//
// Being OP a model of OptionalPointee (either a pointer or an optional):
//
// If both x and y have valid pointees, returns the result of (*x == *y)
// If only one has a valid pointee, returns false.
// If none have valid pointees, returns true.
// No-throw
template <class OptionalPointee>
inline bool equal_pointees(OptionalPointee const& x, OptionalPointee const& y) {
  return (!x) != (!y) ? false : (!x ? true : (*x) == (*y));
}

template <class OptionalPointee>
struct equal_pointees_t
    : std::binary_function<OptionalPointee, OptionalPointee, bool> {
  bool operator()(OptionalPointee const& x, OptionalPointee const& y) const {
    return equal_pointees(x, y);
  }
};

// template<class OP> bool less_pointees(OP const& x, OP const& y);
// template<class OP> struct less_pointees_t;
//
// Being OP a model of OptionalPointee (either a pointer or an optional):
//
// If y has not a valid pointee, returns false.
// ElseIf x has not a valid pointee, returns true.
// ElseIf both x and y have valid pointees, returns the result of (*x < *y)
// No-throw
template <class OptionalPointee>
inline bool less_pointees(OptionalPointee const& x, OptionalPointee const& y) {
  return !y ? false : (!x ? true : (*x) < (*y));
}

template <class OptionalPointee>
struct less_pointees_t
    : std::binary_function<OptionalPointee, OptionalPointee, bool> {
  bool operator()(OptionalPointee const& x, OptionalPointee const& y) const {
    return less_pointees(x, y);
  }
};

}  // namespace paddle

#endif
