## Interaction between C++ and Python

Users employ API in Python to describe their own network, however, the network construction actually happens in C++. so Protobuf is introduced to send the message between Python and C++. 

The Interaction between Python and C++ can be simplified as two steps:

1. C++ tells Python how many Ops there are, and what parameter do users need to offer to initialize a new Op. Python then builds API for each Op at compile time.

2. Users invoke APIs built by Python and provide necessary parameters. These parameters will be sent to C++ fo finish Op construction task.

### Message form C++ to Python

We define a Protobuf message class `OpProto` to hold message needed in the first step. What should an `OpProto` contain? This question is equivalent to “What message do we need to offer, to build a Python API which is legal and user oriented and can use to describe a whole Op.”

Following message are necessary:

1. Op's name, and its simple comment.
2. Input and output variable number; each variable's name, type, and comment.
3. Op's attributes; each attribute includes name, type, comment, **default value** and **value range**.

So `OpProto` can be defined as follows:

```proto
enum AttrType {
	INT = 1;
	FLOAT = 2;
	STRING = 3;
	INTS = 4;
	FLOATS = 5;
	STRINGS = 6;
};

message AttrValue {
	AttrType type = 1;
	optional int iv = 2;
	optional float fv = 3;
	optional string sv = 4;
	repeated int ivs = 5;
	repeated float fvs = 6;
	repeated string svs = 7;
};

message AttrProto {
	required string name = 1;
	required string comment = 2;
	optional AttrValue default = 3;
	optional AttrValue max = 4;
	optional AttrValue min = 5;
	required AttrType type = 6;
};

message VarProto {
	required string name = 1;
	required string comment = 2;
};

message OpProto {
	repeated VarProto inputs = 1;
	repeated VarProto outputs = 2;
	repeated AttrProto attrs = 3;
	required string type = 4;
	required string comment = 5;
};
```

The default value and value range didn't appear in out previous design. By adding these two fields, we are able to check attribute validity in Python and find out possible error as soon as possible. What's more, by providing the message about default value and value range to Python docstring, it helps to automatically generate more comprehensive documents.

### Message from Python to C++

To hold message needed in the above second step, we define Protobuf message class `OpDesc`. It is used to hold user-specified parameters in Op describing.

```proto
message OpDesc {
	required string type = 1;	
	repeated string inputs = 2;
	repeated string outputs = 3;
	map<string, AttrValue> attrs = 4;
};
```

## OpProto Register

Every Op has its own `OpProto`. For using convenience, we need to register them and record all their messages. For each `Op` class, we define a corresponding `OpMaker` class, in whose constructor we implement the `OpProto`'s building process. `OpMaker`'s constructor will be invoked by another function `OpRegistry::RegisterOp()`.

```cpp
class OpProtoMaker {
public:
	OpProtoMaker(OpProto* proto): proto_(proto) {}
protected:
	OpProto* proto_;
	void AddInput(const std::string& name, const std::string& desc) {...}
	void AddAttr(const std::string& name, const std::string& desc, TypeId type) {...}
	void AddComment(const std::string& comment) { ... }
};

class OpRegistry {
public:
	using OpCreator = std::function<OperatorBase* (OpDesc& desc)>;
	
	template <typename OpType, typename OpMaker>
	static void RegisterOp(const std::string& name) {
		gCreators_[name] = [](const OpDesc& desc) {
			return new OpType(desc);
		};
		OpProto& opProto = gProtos_[name];
		OpMaker()(&opProto);
	}

	static map<string, OpCreator> gCreators_;
	static map<string, OpProto> gProtos_;
};

template <typename OpType, typename OpMaker>
class OpRegister {
  public:
    OpRegister(std::string type) {
        OpRegistry::RegisterOp<OpType, OpMaker>(type);
    }
};

#define REGISTER_OP(op_class, op_maker_class, type_name)         \
    class op_class##Register {                                   \
      private:                                                   \
        const static OpRegister<#op_class, #op_maker_class> reg; \
    };                                                           \
    const Register op_class##Register::reg(#type_name);
    
class CosineOp {
// ...
}

struct CosineOpProtoMaker : public OpProtoMaker {
	CosineOpProtoMaker(OpProto* proto) : OpProtoMaker(proto) {
		AddInput("input", "input of cosine op");
		AddAttr("scale", "scale of cosine op", float).Default(1.0).LargerThan(0.0);
		AddType("cos");
		AddComment("This is cos op");
	}
}

REGISTER_OP(CosineOp, CosineOpProtoMaker, cos);
```

In `REGISTER_OP(CosineOp, CosineOpProtoMaker, cos)`, we register not only `CosineOp` but also `CosineOpProto`. As fields of `CosineOpProto`, the default value and value range of `scale` are also registered here. 

## Python API

Python  APIs are divided into two types, high-level API and low-level API.

### High-Level API

High-level API is called by users directly, so it should keep its style consistent with existing V2 APIs.

Here is a sample about how a define a fc layer:

```python
hd = fc_layer(input=data, size=56, with_bias=True, activation="sigmoid");
```

`hd` is the output of `fc_layer` and it's a `variable`. It can be further sent into other layers as input.

The definition of `fc_layer()`:

```python
def fc_layer(input, size, with_bias, activation):
	attr_map = {"size":size}
	check_attrs(attr_map)
	w = make_variable('w')
	if with_bias:
		b = make_variable('b')
	else:
		b = None
	fc_output = make_variable('fc_output');
	fc_op(input, w, b, fc_output, attr_map)
	act_output = make_variable('sigmod_output');
	if activation == "sigmod":
		sigmod_op(fc_output, act_output);
	elif:
		# ...
	return act_output;
``` 

### Low Leval API

In above sample, `fc_op` and `sigmod_op` are low-level API. They build `OpDesc` and invoke corresponding C++ code.

*TODO*

## Op and Kernal

After completely defined, an Op will be run in a network. However, Op's computing method may differ on different devices. One solution is that write an `Op`'s member function `Op::run()`, which contains computing methods of all possible devices. That may be a bad idea because we have to change all `Op`'s code to add a new device.

Another choice is adding a concept named `kernal`. A `Kernal` describes an op's computing process on a certain device. After stripping `Variable` and `kernal`, `Op` becomes a pure conceptual class, which holds neither data nor detailed computing process.

```cpp
class KernalBase {
public:
  virtual void RunOnDevice(std::vector<Variable*> input_vars,
                           std::vector<Variable*> input_vars,
                           const OpAttrs* attrs) = 0;  
};

template <typename Device>
class CosineKernal : public KernalBase {
public:
  virtual void RunOnDevice(std::vector<Variable*> input_vars,
                           std::vector<Variable*> input_vars,
                           const OpAttrs* attrs) {
    // no implementation
  }
};

template <>
class CosineKernal<CpuDevice> : public KernalBase {
public:
  virtual void RunOnDevice(std::vector<Variable*> input_vars,
                           std::vector<Variable*> input_vars,
                           const OpAttrs* attrs) {
    CosineOpAttrs* cosine_attrs = static_cast<CosineOpAttrs*>(attrs);
    // computing code
    // ...
  }
};

struct OpAttrs {...};

class Op {
 public:
   std::string get_kernal_name() {
     return kernel_name_;
   }
   const vector<std::string>& get_input_names() {
     return input_names_;
    }
   const vector<std::string>& get_output_names() {
     return output_names_;
   }
 // ...
 private:
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string kernal_name_;
  
}

struct CosineOpAttrs : public OpAttrs {
  float scale_;
}
  
class CosineOp : public Op {
 public:
  const CosineOpAtrrs* get_attrs() {
    return &attrs;
  }
  
 private:
  CosineOpAttrs attrs;
}

RunOp(const Op& op, Scope scope) {
  Kernal* kernal = get_kernal(scope, op.get_kernal_name());
  std::vector<Variable*> input_vars = 
               get_variables(scope, op.get_input_name());
  std::vector<Variable*> output_vars = 
               get_variables(scope, op.get_output_name());
  	  
  kernal->RunOnDevice(input_vars, output_vars, op.get_attrs());
}
```

All `Kernal` need to be registered beforehand, just like `Op`.

Now, `Op` is no longer has `Run()` function. It only contains names of variables and kernels. During network running, `RunOp()` is called to invoke `Op`'s corresponding `Kernal`. `get_kernal()` is supposed to return `kernal` for current device.
