#include "paddle/fluid/inference/analysis/dot.h"

#include <gtest/gtest.h>
#include <memory>
#include "paddle/fluid/inference/analysis/data_flow_graph.h"

namespace paddle {
namespace inference {
namespace analysis {

class DotTester : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<Dot::Attr> attrs({{"title", "hello"}});
    dot.reset(new Dot(attrs));
    dot->AddNode("a", {Dot::Attr{"shape", "box"}, Dot::Attr("color", "blue")});
    dot->AddNode("b", {});
    dot->AddNode("c", {});
    dot->AddEdge("a", "b", {});
    dot->AddEdge("b", "c", {});
    dot->AddEdge("a", "c", {});
  }

  std::unique_ptr<Dot> dot;
};

TEST_F(DotTester, Build) {
  auto codes = dot->Build();
  LOG(INFO) << '\n' << codes;
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
