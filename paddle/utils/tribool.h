// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file copy from boost/logic/tribool.hpp, boost version: 1.41.0
// Modified the following points:
// 1. modify namespace from boost to paddle
// 2. remove the depending boost header files
// 3. remove the dummy_ in indeterminate_t, which is specially implemented for
// Borland C++ Builder
// 4. remove unnecessary macro BOOST_TRIBOOL_THIRD_STATE

// Three-state boolean logic library

// Copyright Douglas Gregor 2002-2004. Use, modification and
// distribution is subject to the Boost Software License, Version
// 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// For more information, see http://www.boost.org

#pragma once

namespace paddle {
namespace logic {

/// INTERNAL ONLY
namespace detail {
/**
 * INTERNAL ONLY
 *
 * \brief A type used only to uniquely identify the 'indeterminate'
 * function/keyword.
 */
struct indeterminate_t {};

}  // end namespace detail

class tribool;

/**
 * INTERNAL ONLY
 * The type of the 'indeterminate' keyword. This has the same type as the
 * function 'indeterminate' so that we can recognize when the keyword is
 * used.
 */
typedef bool (*indeterminate_keyword_t)(tribool, detail::indeterminate_t);

/**
 * \brief Keyword and test function for the indeterminate tribool value
 *
 * The \c indeterminate function has a dual role. It's first role is
 * as a unary function that tells whether the tribool value is in the
 * "indeterminate" state. It's second role is as a keyword
 * representing the indeterminate (just like "true" and "false"
 * represent the true and false states).
 *
 * \returns <tt>x.value == tribool::indeterminate_value</tt>
 * \throws nothrow
 */
inline bool indeterminate(
    tribool x, detail::indeterminate_t dummy = detail::indeterminate_t());

/**
 * \brief A 3-state boolean type.
 *
 * 3-state boolean values are either true, false, or
 * indeterminate.
 */
class tribool {
 private:
  /// INTERNAL ONLY
  struct dummy {
    void nonnull() {}
  };

  typedef void (dummy::*safe_bool)();

 public:
  /**
   * Construct a new 3-state boolean value with the value 'false'.
   *
   * \throws nothrow
   */
  tribool() : value(false_value) {}

  /**
   * Construct a new 3-state boolean value with the given boolean
   * value, which may be \c true or \c false.
   *
   * \throws nothrow
   */
  tribool(bool value) : value(value ? true_value : false_value) {}  // NOLINT

  /**
   * Construct a new 3-state boolean value with an indeterminate value.
   *
   * \throws nothrow
   */
  tribool(indeterminate_keyword_t) : value(indeterminate_value) {}  // NOLINT

  /**
   * Use a 3-state boolean in a boolean context. Will evaluate true in a
   * boolean context only when the 3-state boolean is definitely true.
   *
   * \returns true if the 3-state boolean is true, false otherwise
   * \throws nothrow
   */
  operator safe_bool() const {
    return value == true_value ? &dummy::nonnull : 0;
  }

  /**
   * The actual stored value in this 3-state boolean, which may be false, true,
   * or indeterminate.
   */
  enum value_t { false_value, true_value, indeterminate_value } value;
};

// Check if the given tribool has an indeterminate value. Also doubles as a
// keyword for the 'indeterminate' value
inline bool indeterminate(tribool x, detail::indeterminate_t) {
  return x.value == tribool::indeterminate_value;
}

/** @defgroup logical Logical operations
 */
//@{
/**
 * \brief Computes the logical negation of a tribool
 *
 * \returns the logical negation of the tribool, according to the
 * table:
 *  <table border=1>
 *    <tr>
 *      <th><center><code>!</code></center></th>
 *      <th/>
 *    </tr>
 *    <tr>
 *      <th><center>false</center></th>
 *      <td><center>true</center></td>
 *    </tr>
 *    <tr>
 *      <th><center>true</center></th>
 *      <td><center>false</center></td>
 *    </tr>
 *    <tr>
 *      <th><center>indeterminate</center></th>
 *      <td><center>indeterminate</center></td>
 *    </tr>
 *  </table>
 * \throws nothrow
 */
inline tribool operator!(tribool x) {
  return x.value == tribool::false_value  ? tribool(true)
         : x.value == tribool::true_value ? tribool(false)
                                          : tribool(indeterminate);
}

/**
 * \brief Computes the logical conjuction of two tribools
 *
 * \returns the result of logically ANDing the two tribool values,
 * according to the following table:
 *       <table border=1>
 *           <tr>
 *             <th><center><code>&amp;&amp;</code></center></th>
 *             <th><center>false</center></th>
 *             <th><center>true</center></th>
 *             <th><center>indeterminate</center></th>
 *           </tr>
 *           <tr>
 *             <th><center>false</center></th>
 *             <td><center>false</center></td>
 *             <td><center>false</center></td>
 *             <td><center>false</center></td>
 *           </tr>
 *           <tr>
 *             <th><center>true</center></th>
 *             <td><center>false</center></td>
 *             <td><center>true</center></td>
 *             <td><center>indeterminate</center></td>
 *           </tr>
 *           <tr>
 *             <th><center>indeterminate</center></th>
 *             <td><center>false</center></td>
 *             <td><center>indeterminate</center></td>
 *             <td><center>indeterminate</center></td>
 *           </tr>
 *       </table>
 * \throws nothrow
 */
inline tribool operator&&(tribool x, tribool y) {
  if (static_cast<bool>(!x) || static_cast<bool>(!y))
    return false;
  else if (static_cast<bool>(x) && static_cast<bool>(y))
    return true;
  else
    return indeterminate;
}

/**
 * \overload
 */
inline tribool operator&&(tribool x, bool y) { return y ? x : tribool(false); }

/**
 * \overload
 */
inline tribool operator&&(bool x, tribool y) { return x ? y : tribool(false); }

/**
 * \overload
 */
inline tribool operator&&(indeterminate_keyword_t, tribool x) {
  return !x ? tribool(false) : tribool(indeterminate);
}

/**
 * \overload
 */
inline tribool operator&&(tribool x, indeterminate_keyword_t) {
  return !x ? tribool(false) : tribool(indeterminate);
}

/**
 * \brief Computes the logical disjunction of two tribools
 *
 * \returns the result of logically ORing the two tribool values,
 * according to the following table:
 *       <table border=1>
 *           <tr>
 *             <th><center><code>||</code></center></th>
 *             <th><center>false</center></th>
 *             <th><center>true</center></th>
 *             <th><center>indeterminate</center></th>
 *           </tr>
 *           <tr>
 *             <th><center>false</center></th>
 *             <td><center>false</center></td>
 *             <td><center>true</center></td>
 *             <td><center>indeterminate</center></td>
 *           </tr>
 *           <tr>
 *             <th><center>true</center></th>
 *             <td><center>true</center></td>
 *             <td><center>true</center></td>
 *             <td><center>true</center></td>
 *           </tr>
 *           <tr>
 *             <th><center>indeterminate</center></th>
 *             <td><center>indeterminate</center></td>
 *             <td><center>true</center></td>
 *             <td><center>indeterminate</center></td>
 *           </tr>
 *       </table>
 *  \throws nothrow
 */
inline tribool operator||(tribool x, tribool y) {
  if (static_cast<bool>(!x) && static_cast<bool>(!y))
    return false;
  else if (static_cast<bool>(x) || static_cast<bool>(y))
    return true;
  else
    return indeterminate;
}

/**
 * \overload
 */
inline tribool operator||(tribool x, bool y) { return y ? tribool(true) : x; }

/**
 * \overload
 */
inline tribool operator||(bool x, tribool y) { return x ? tribool(true) : y; }

/**
 * \overload
 */
inline tribool operator||(indeterminate_keyword_t, tribool x) {
  return x ? tribool(true) : tribool(indeterminate);
}

/**
 * \overload
 */
inline tribool operator||(tribool x, indeterminate_keyword_t) {
  return x ? tribool(true) : tribool(indeterminate);
}
//@}

/**
 * \brief Compare tribools for equality
 *
 * \returns the result of comparing two tribool values, according to
 * the following table:
 *       <table border=1>
 *          <tr>
 *            <th><center><code>==</code></center></th>
 *            <th><center>false</center></th>
 *            <th><center>true</center></th>
 *            <th><center>indeterminate</center></th>
 *          </tr>
 *          <tr>
 *            <th><center>false</center></th>
 *            <td><center>true</center></td>
 *            <td><center>false</center></td>
 *            <td><center>indeterminate</center></td>
 *          </tr>
 *          <tr>
 *            <th><center>true</center></th>
 *            <td><center>false</center></td>
 *            <td><center>true</center></td>
 *            <td><center>indeterminate</center></td>
 *          </tr>
 *          <tr>
 *            <th><center>indeterminate</center></th>
 *            <td><center>indeterminate</center></td>
 *            <td><center>indeterminate</center></td>
 *            <td><center>indeterminate</center></td>
 *          </tr>
 *      </table>
 * \throws nothrow
 */
inline tribool operator==(tribool x, tribool y) {
  if (indeterminate(x) || indeterminate(y))
    return indeterminate;
  else
    return (x && y) || (!x && !y);
}

/**
 * \overload
 */
inline tribool operator==(tribool x, bool y) { return x == tribool(y); }

/**
 * \overload
 */
inline tribool operator==(bool x, tribool y) { return tribool(x) == y; }

/**
 * \overload
 */
inline tribool operator==(indeterminate_keyword_t, tribool x) {
  return tribool(indeterminate) == x;
}

/**
 * \overload
 */
inline tribool operator==(tribool x, indeterminate_keyword_t) {
  return tribool(indeterminate) == x;
}

/**
 * \brief Compare tribools for inequality
 *
 * \returns the result of comparing two tribool values for inequality,
 * according to the following table:
 *       <table border=1>
 *           <tr>
 *             <th><center><code>!=</code></center></th>
 *             <th><center>false</center></th>
 *             <th><center>true</center></th>
 *             <th><center>indeterminate</center></th>
 *           </tr>
 *           <tr>
 *             <th><center>false</center></th>
 *             <td><center>false</center></td>
 *             <td><center>true</center></td>
 *             <td><center>indeterminate</center></td>
 *           </tr>
 *           <tr>
 *             <th><center>true</center></th>
 *             <td><center>true</center></td>
 *             <td><center>false</center></td>
 *             <td><center>indeterminate</center></td>
 *           </tr>
 *           <tr>
 *             <th><center>indeterminate</center></th>
 *             <td><center>indeterminate</center></td>
 *             <td><center>indeterminate</center></td>
 *             <td><center>indeterminate</center></td>
 *           </tr>
 *       </table>
 * \throws nothrow
 */
inline tribool operator!=(tribool x, tribool y) {
  if (indeterminate(x) || indeterminate(y))
    return indeterminate;
  else
    return !((x && y) || (!x && !y));
}

/**
 * \overload
 */
inline tribool operator!=(tribool x, bool y) { return x != tribool(y); }

/**
 * \overload
 */
inline tribool operator!=(bool x, tribool y) { return tribool(x) != y; }

/**
 * \overload
 */
inline tribool operator!=(indeterminate_keyword_t, tribool x) {
  return tribool(indeterminate) != x;
}

/**
 * \overload
 */
inline tribool operator!=(tribool x, indeterminate_keyword_t) {
  return x != tribool(indeterminate);
}

}  // namespace logic
}  // namespace paddle

// Pull tribool and indeterminate into namespace "boost"
namespace paddle {
using logic::indeterminate;
using logic::tribool;
}  // namespace paddle
