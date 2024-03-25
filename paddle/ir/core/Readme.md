## Language Reference

### Remark and Grammer

See [wiki EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form).

### ModuleOp/Program

```
Program ::=  [ ParameterList ] ModuleOp;
ModuleOp ::=  { Region };

ParameterList ::= { Parameter };
Parameter ::= Stringidentifier ":" Type "\n";
```

### Region/Block

```
Region ::= { Block };
Block ::= "{" { Operation } "}" ;
```

### Operation
```
Operation                 ::= OpResultList? "=" (GenericOperation | CustomOperation)
GenericOperation          ::= OpName "(" OperandList? ")"  AttributeMap ":" FunctionType
OpName                    ::= "\"" StringIdentifier "." StringIdentifier "\""
CustomOperation           ::= CustomOperationFormat
OpResultList              ::= ValueList
OperandList               ::= ValueList
ValueList                 ::= ValueId ("," ValueId)*
ValueId                   ::= "%" Digits
AttributeMap              ::= "{" (AttributeEntry ("," AttributeEntry)* ) "}"
AttributeEntry            ::= StringIdentifier ":" Attribute
FunctionType              ::= TypeList '->' TypeList
TypeList                  ::= Type (",", Type)*
```

### Type/Attribute
```
Type ::= StringIdentifier
Attribute ::= StringIdentifier
```

The `StringIdentifier` is still need to be decided, considering which symbols will be used as `keyword` .
