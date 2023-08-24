#pragma once
#include <istream>
#include "paddle/ir/core/parser/token.h"

enum LexSegment {
  parseOpResult = 0,
  parseOpInfo = 1,
  parseOpRand = 2,
  parserAttribute = 3,
  parseFunctionType = 4,
};

class Lexer {
 private:
  std::istream& is;
  size_t line = 1;
  size_t column = 1;

 public:
  explicit Lexer(std::istream& is) : is(is) {}
  ~Lexer() = default;
  Token GetToken(LexSegment seg);
  Token* LexIdentifer(LexSegment seg);
  Token* LexNumberOrArraow();
  Token* LexEndTagOrNullVal(LexSegment seg);
  Token* LexValueId();
  Token* LexEOF();
  Token* LexOpName();
  char GetChar();
  void SkipWhitespace();
  bool IsEndTag(char, LexSegment);
  bool IsSpace(char);
  size_t GetLine();
  size_t GetColumn();
};
