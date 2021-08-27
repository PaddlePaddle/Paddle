// This file copy from boost/none_t.hpp and boost/none.hpp and boost version:
// 1.41.0
// Modified the following points:
// 1. modify namespace from boost::none to paddle::none
// 2. modify namespace from boost::none_t to paddle::none_t

// Copyright (C) 2003, Fernando Luis Cacciola Carballal.
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org/libs/optional for documentation.
//
// You are welcome to contact the author at:
//  fernando_cacciola@hotmail.com
//
#ifndef PADDLE_NONE_17SEP2003_HPP
#define PADDLE_NONE_17SEP2003_HPP

namespace paddle {

namespace detail {
struct none_helper {};
}

typedef int detail::none_helper::*none_t;

}  // namespace boost

// NOTE: Borland users have to include this header outside any precompiled
// headers
// (bcc<=5.64 cannot include instance data in a precompiled header)
//  -- * To be verified, now that there's no unnamed namespace

namespace paddle {

none_t const none = ((none_t)0);

}  // namespace boost

#endif
