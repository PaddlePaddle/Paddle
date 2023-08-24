#pragma once
#include <string>
using std::string;

enum Token_type {
  EOF_ = -1,
  IDENTIFER = 0,
  DIGIT = 1,
  SDIGIT = 2,
  ENDTAG = 3,
  VALUEID = 4,
  OPNAME = 5,
  ARRAOW = 6,
  NULL_ = 7,
};

class Token {
 public:
  string val_;
  Token_type token_type_;
  Token() = default;
  Token(string val, Token_type token_type) {
    val_ = val;
    token_type_ = token_type;
  }
};
