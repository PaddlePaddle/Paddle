# DRR (Declarative Rewrite Rule) Tool User Manual
---
## 1. Related Background

PASS is a key component for optimizing IR, and the transformation of DAG-to-DAG (replacing one DAG subgraph in the original image with another DAG subgraph) is the most common type of Pass. The transformation of DAG-to-DAG can be divided into two steps: matching and rewriting. Matching refers to the complete matching of a known subgraph to the corresponding target subgraph in the Program, while rewriting refers to replacing the matched graph structure with a new subgraph.

DRR (Declarative Rewrite Rule) is a set of PASS components used to handle this type of DAG-to-DAG. DRR can reduce the development cost of PASS, allowing developers to focus on optimizing logic processing without worrying about the underlying IR data structure. After the developer declares the target subgraph and the new subgraph that needs to be replaced with a pattern through a simple and easy-to-use interface, DRR can automatically match the original graph in the Program and replace it with a new subgraph.

Taking the PASS of eliminating redundant CastOp as an example, the code development example using DRR is as follows:
~~~ c++
// 1. Inherit specialized template classes from DrPatternBase
class RemoveRedundentCastPattern
    : public pir::drr::DrrPatternBase<RemoveRedundentCastPattern> {
  // 2. Overload operator()
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // 3. Define a SourcePattern containing two consecutive CastOps using Op, Tensor, and Attribute
    auto pat = ctx->SourcePattern();

    pat.Tensor("tmp") =                          // CastOp output Tensor named "tmp"
        pat.Op(paddle::dialect::CastOp::name(),  // Name passed in to CastOp
               {{"dtype", pat.Attr("dtype1")}})  // The global unique ID corresponding to the "dtype" attribute of CastOp is "dtype1"
               (pat.Tensor("arg0"));             // CastOp input Tensor is "arg0 "
    pat.Tensor("ret") =
        pat.Op(paddle::dialect::CastOp::name(),
               {{"dtype", pat.Attr("dtype2")}})(pat.Tensor("tmp"));
    // 4. Define Constrain
    pat.RequireEqual(pat("tmp").dtype(), pat.Tensor("ret").dtype());

    // 5. Define ResultPattern
    auto res = pat.ResultPattern();
    res.Tensor("ret") =
        res.Op(paddle::dialect::CastOp::name(),
               {{"dtype", pat.Attr("dtype2")}})(res.Tensor("arg0"));
  }
};
~~~

DRR PASS consists of the following three parts::
+ `Source Pattern`：Used to describe the target subgraph to be matched in the Program
+ `Result Pattern`：Constraints used to specify SourcePattern matching (nonessential)
+ `Constrains`：Used to describe the pattern subgraph that needs to be replaced with, developers only need to define the SourcePattern, Constrains, and ResultPattern to implement a complete PASS.

**Note:**
1. **DRR only supports matching and replacing the SourcePattern and ResultPattern of the closure (except for the Pattern input and output Tensors, all internal Tensors cannot be used by external Ops of the Pattern). If the defined Pattern is not closed in the Program, matching fails**
2. **The input and output of ResultPattern need to be a subset of the input and output of SourcePattern**
## 2. Interface List
<table>
	 <tr>
		<th> Class </th>
		<th> Function </th>
		<th> Function Description </th>
		<th> Parameter Interpretation </th>
	 </tr>
	<tr>
		<td rowspan="1">DrrPatternBase</td>
		<td> <pre> virtual void operator()(
        pir::drr::DrrPatternContext* ctx) const </pre></td>
		<td> Implement the entry function of DRR PASS </td>
		<td> ctx: Context parameters required to create Patten</td>
	</tr>
	<tr>
		<td rowspan="6"> SourcePattern</td>
		<td><pre> const drr::Op& Op(
    const std::string& op_type,
    const std::unordered_map&lt;std::string, Attribute&gt;& attributes)</pre></td>
		<td> Define an Op in the SourcePattern</td>
		<td> op_type: The defined Op name can be obtained through the paddle::dialect::xxOp
	::name() interface <br> attributes : Attribute information of the created Op </td>
	</tr>
	<tr>
		<td><pre> const drr::Tensor& Tensor(
        const std::string& tensor_name) </pre></td>
		<td> Define a Tensor in the SourcePattern_ Tensor of name</td>
		<td>  tensor_name: The name of the defined Tensor must be unique within the SourcePattern </td>
	</tr>
	<tr>
		<td> <pre> Attribute Attr(
        const std::string& attr_name) const </pre></td>
		<td> Define an attr named in the SourcePattern_ Properties of name </td>
		<td> attr_name: The name of the attribute must be unique within the SourcePattern </td>
	</tr>
	<tr>
		<td><pre> void RequireEqual(
        const TensorShape& first,
        const TensorShape& second)</pre></td>
		<td> Require the TensorShapes of two Tensors in the SourcePattern to be the same</td>
		<td> first: first TensorShape <br> second : second TensorShape</td>
	</tr>
		<tr>
		<td><pre> void RequireEqual(
        const TensorDataType& first,
        const TensorDataType& second)</pre></td>
		<td> Require two Tensors in the SourcePattern to have the same data type</td>
		<td> first: DataType of the first Tensor <br> second : DataType of the second Tensor</td>
	</tr>
	<tr>
		<td> <pre>void RequireNativeCall(
        const std::function&lt;bool(const MatchContext&)&gt;& custom_fn)</pre></td>
		<td> Define a constraint in SourcePattern, which can be used to implement custom constraints on SourcePattern using this interface and lamda expressions</td>
		<td> custom_fn: Customized constraint functions</td>
	</tr>
	<tr>
		<td rowspan="5"> ResultPattern</td>
				<td><pre> const drr::Op& Op(
    const std::string& op_type,
    const std::unordered_map&lt;std::string, Attribute&gt;&  attributes) </pre></td>
		<td> Define an Op in ResultPattern </td>
		<td> op_type: The defined Op name can be obtained through the paddle::dialect::xxOp
	::name() interface<br> attributes : Attribute information of the created Op </td>
	</tr>
	<tr>
		<td> <pre>const drr::Tensor& Tensor(
        const std::string& tensor_name)</pre></td>
		<td> Define a tensor named tensor_name in ResultPattern</td>
		<td> tensor_name: The name of the defined Tensor must be unique within the ResultPattern </td>
	</tr>
	<tr>
		<td><pre>Attribute Attr(
        const std::string& attr_name) const </pre></td>
		<td> Define an attribute named attr_name in ResultPattern </td>
		<td> attr_name: The name of the attribute must be unique within the ResultPattern </td>
	</tr>
<tr>
		<td><pre>using AttrComputeFunc = std::function&lt;std::any(const MatchContext&)&gt;;
Attribute Attr(const AttrComputeFunc& attr_compute_func) const</pre></td>
		<td> Create an Attribute through a custom calculation logic AttrComputeFunc</td>
		<td>attr_compute_func: Customized calculation logic</td>
	</tr>
	<tr>
		<td> <pre>drr::Tensor& NoneTensor()</pre></td>
		<td> When the input Tensor of an Op is optional and not needed, NoneTensior needs to be used to hold the position</td>
		<td> / </td>
	</tr>
	<tr>
		<td rowspan="2"> TensorShape</td>
		<td><pre>explicit TensorShape(
        const std::string& tensor_name) </pre></td>
		<td> Abstract the class that describes the shape of Tensor </td>
		<td> tensor_name: The name of the Tensor being described </td>
	</tr>
	<tr>
		<td><pre> const std::string& tensor_name() const</pre></td>
		<td> Obtain the name of Tensor </td>
		<td>  / </td>
	</tr>
	<tr>
		<td rowspan="2"> TensorDataType</td>
		<td><pre>explicit TensorDataType(
        const std::string& tensor_name)</pre></td>
		<td> An abstract class that describes the data types of elements in Tensor </td>
		<td> tensor_name: The name of the Tensor being described </td>
	</tr>
	<tr>
		<td><pre> const std::string& tensor_name() const</pre></td>
		<td> Obtain the name of Tensor </td>
		<td> / </td>
	</tr>
	<tr>
		<td rowspan="1"> DrrPatternContext</td>
		<td><pre>drr::SourcePattern DrrPatternContext::SourcePattern()</pre> </td>
		<td> Create a SourcePattern object and return </td>
		<td> / </td>
	</tr>
</table>

## 3 Example
Example 1: Matmul + Add -> FusedGemmEpilogue
~~~ c++
class FusedLinearPattern : public pir::drr::DrrPatternBase<FusedLinearPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
	// Declare SourcePattern
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));

    // Declare ResultPattern
    pir::drr::ResultPattern res = pat.ResultPattern();
    // Declare Constrain
    const auto &act_attr =
        res.Attr([](const pir::drr::MatchContext &match_ctx) -> std::any {
          return "none";
        });
    const auto &fused_gemm_epilogue = res.Op(paddle::dialect::FusedGemmEpilogueOp::name(),
                                             {{{"trans_x", pat.Attr("trans_x")},
                                               {"trans_y", pat.Attr("trans_y")},
                                               {"activation", act_attr}}});
    fused_gemm_epilogue(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out")});
  }
};
~~~

Example 2: Full + Expand -> Full
~~~ c++
class FoldExpandToConstantPattern
    : public pir::drr::DrrPatternBase<FoldExpandToConstantPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Declare SourcePattern
    pir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &full1 = pat.Op(paddle::dialect::FullOp::name(),
                               {{"shape", pat.Attr("shape_1")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    const auto &full_int_array1 =
        pat.Op(paddle::dialect::FullIntArrayOp::name(),
               {{"value", pat.Attr("expand_shape_value")},
                {"dtype", pat.Attr("dtype_2")},
                {"place", pat.Attr("place_2")}});
    const auto &expand = pat.Op(paddle::dialect::ExpandOp::name());
    pat.Tensor("ret") = expand(full1(), full_int_array1());

    // Declare ResultPattern
    pir::drr::ResultPattern res = pat.ResultPattern();
    const auto &full2 = res.Op(paddle::dialect::FullOp::name(),
                               {{"shape", pat.Attr("expand_shape_value")},
                                {"value", pat.Attr("value_1")},
                                {"dtype", pat.Attr("dtype_1")},
                                {"place", pat.Attr("place_1")}});
    res.Tensor("ret") = full2();
  }
};
~~~
