#include <iostream>
#include <sstream>  // std::stringstream
#include <string>

#include "paddle/framework/op_info.h"
#include "paddle/framework/op_registry.h"
#include "paddle/pybind/pybind.h"

std::string Escape(const std::string& s) {
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
      case '\t':
        r += "\\t";
      case '\r':
        break;
      default:
        r += s[i];
        break;
    }
  }
  return r;
}

std::string AttrType(paddle::framework::AttrType at) {
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

void PrintVar(const paddle::framework::OpProto::Var& v, std::stringstream& ss) {
  ss << " { "
     << "\n"
     << "   \"name\" : \"" << Escape(v.name()) << "\",\n"
     << "   \"comment\" : \"" << Escape(v.comment()) << "\",\n"
     << "   \"duplicable\" : " << v.duplicable() << ",\n"
     << "   \"intermediate\" : " << v.intermediate() << "\n"
     << " },";
}

void PrintAttr(const paddle::framework::OpProto::Attr& a,
               std::stringstream& ss) {
  ss << " { "
     << "\n"
     << "   \"name\" : \"" << Escape(a.name()) << "\",\n"
     << "   \"type\" : \"" << AttrType(a.type()) << "\",\n"
     << "   \"comment\" : \"" << Escape(a.comment()) << "\",\n"
     << "   \"generated\" : " << a.generated() << "\n"
     << " },";
}

void PrintOpProto(const std::string& type,
                  const paddle::framework::OpInfo& opinfo,
                  std::stringstream& ss) {
  std::cerr << "Processing " << type << "\n";

  const paddle::framework::OpProto* p = opinfo.proto_;
  if (p == nullptr) {
    return;  // It is possible that an operator doesn't have OpProto.
  }

  ss << "{\n"
     << " \"type\" : \"" << Escape(p->type()) << "\",\n"
     << " \"comment\" : \"" << Escape(p->comment()) << "\",\n";

  ss << " \"inputs\" : [ "
     << "\n";
  for (int i = 0; i < p->inputs_size(); i++) {
    PrintVar(p->inputs(i), ss);
  }
  ss.seekp(-1, ss.cur);  // remove the trailing comma
  ss << " ], "
     << "\n";

  ss << " \"outputs\" : [ "
     << "\n";
  for (int i = 0; i < p->outputs_size(); i++) {
    PrintVar(p->outputs(i), ss);
  }
  ss.seekp(-1, ss.cur);  // remove the trailing comma
  ss << " ], "
     << "\n";

  ss << " \"attrs\" : [ "
     << "\n";
  for (int i = 0; i < p->attrs_size(); i++) {
    PrintAttr(p->attrs(i), ss);
  }
  ss.seekp(-1, ss.cur);  // remove the trailing comma
  ss << " ] "
     << "\n";

  ss << "},";
}

int main() {
  std::stringstream ss;
  ss << "[\n";
  for (auto& iter : paddle::framework::OpInfoMap::Instance().map()) {
    PrintOpProto(iter.first, iter.second, ss);
  }
  ss.seekp(-1, ss.cur);  // remove the trailing comma
  ss << "]\n";
  std::cout << ss.str();
}
