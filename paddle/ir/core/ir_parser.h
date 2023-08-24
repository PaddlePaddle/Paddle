#pragma once
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/parser/lexer.h"
#include "paddle/ir/core/program.h"

using OpResultMap = std::map<string, ir::OpResult>;
using AttributeMap = std::unordered_map<std::string, ir::Attribute>;
using OpAttributeInfoMap = std::map<string, string>;

namespace ir {
class IrParser {
 public:
  Lexer* lexer;
  IrContext* ctx;
  OpResultMap opresultmap;
  Builder* builder;
  Token last_token;
  Token cur_token;
  LexSegment lex_seg;

 public:
  IrParser(IrContext* ctx, std::istream& is);

  ~IrParser();

  Token GetToken();

  Token PeekToken();

  std::unique_ptr<Program> ParseProgram();

  std::unique_ptr<Region> ParseRegion();

  void ParseBlock(Block& block);  // NOLINT

  Operation* ParseOperation();

  OpInfo ParseOpInfo();

  std::vector<std::string> ParseOpResultIndex();

  std::vector<OpResult> ParseOpRandList();

  AttributeMap ParseAttributeMap();

  std::vector<Type> ParseFunctionTypeList();

  OpResult GetNullValue();

  void NextLexSegment();

  Type ParseType();

  Attribute ParseAttribute();

  string GetErrorLocationInfo();

  void ConsumeAToken(string expect_token_val);
};

}  // namespace ir
