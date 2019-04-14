namespace paddle {
namespace lite {
namespace mir {

class Node {
 public:
  // Tell is instruction.
  bool IsInstruct() const;
  // Tell is an argument.
  bool IsArgument() const;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle