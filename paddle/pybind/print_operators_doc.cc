#include <iostream>
#include <string>

#include "paddle/framework/op_info.h"
#include "paddle/framework/op_registry.h"
#include "paddle/pybind/pybind.h"

void PrintOpProto(const std::string& type,
                  const paddle::framework::OpInfo& opinfo) const {
  std::cerr << "Processing " << type << "\n";

  const paddle::framework::OpProto* p = opinfo.proto_;
  if (p == nullptr) {
    return;  // It is possible that an operator doesn't have OpProto.
  }

  std::cout << "{\n"
            << " \"type\" : \"" << Escape(p->type()) << "\",\n"
            << " \"comment\" : \"" << Escape(p->comment()) << "\",\n";

  std::cout << " \"inputs\" : [ "
            << "\n";
  for (int i = 0; i < p->inputs_size(); i++) {
    PrintVar(p->inputs(i), i < p->inputs_size() - 1);
  }
  std::cout << " ], "
            << "\n";

  std::cout << " \"outputs\" : [ "
            << "\n";
  for (int i = 0; i < p->outputs_size(); i++) {
    PrintVar(p->outputs(i), i < p->outputs_size() - 1);
  }
  std::cout << " ], "
            << "\n";

  std::cout << " \"attrs\" : [ "
            << "\n";
  for (int i = 0; i < p->attrs_size(); i++) {
    PrintAttr(p->attrs(i), i < p->attrs_size() - 1);
  }
  std::cout << " ] "
            << "\n";

  std::cout << "},\n";
}

void PrintVar(const paddle::framework::OpProto::Var& v, bool comma) const {
  std::cout << " { "
            << "\n"
            << "   \"name\" : \"" << Escape(v.name()) << "\",\n"
            << "   \"comment\" : \"" << Escape(v.comment()) << "\",\n"
            << "   \"duplicable\" : " << v.duplicable() << ",\n"
            << "   \"intermediate\" : " << v.intermediate() << "\n"
            << " } " << (comma ? ",\n" : "\n");
}

void PrintAttr(const paddle::framework::OpProto::Attr& a, bool comma) const {
  std::cout << " { "
            << "\n"
            << "   \"name\" : \"" << Escape(a.name()) << "\",\n"
            << "   \"type\" : \"" << AttrType(a.type()) << "\",\n"
            << "   \"comment\" : \"" << Escape(a.comment()) << "\",\n"
            << "   \"generated\" : " << a.generated() << "\n"
            << " } " << (comma ? ",\n" : "\n");
}

std::string AttrType(paddle::framework::AttrType at) const {
  switch (at) {
    case paddle::framework::INT:
      return "int";
    case paddle::framework::FLOAT:
      return "float";
    case paddle::framework::STRING:
      return "string";
    case paddle::framework::BOOLEAN:
      return "bool";
    case paddle::framework::INTS:
      return "int array";
    case paddle::framework::FLOATS:
      return "float array";
    case paddle::framework::STRINGS:
      return "string array";
    case paddle::framework::BOOLEANS:
      return "bool array";
    case paddle::framework::BLOCK:
      return "block id";
  }
  return "UNKNOWN";  // not possible
}

std::string Escape(const std::string& s) const {
  std::string r;
  for (size_t i = 0; i < s.size(); i++) {
    switch (s[i]) {
      case '\"':
        r += "\\\"";
        break;
      case '\\':
        r += "\\\\";
        break;
      case '\n':
        r += "\\n";
        break;
      case '\r':
        break;
      default:
        r += s[i];
        break;
    }
  }
  return r;
}

int main() {
  OpInfoIterator iter;
  std::cout << "[\n";
  for (auto iter : OpInfoMap::Instance().map()) {
    PrintOpProto(iter.first, iter.second);
  }
  std::cout << "]\n";
}
