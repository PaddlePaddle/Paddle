sed -i 's/bool operator ()(const FieldDescriptor\* f1, const FieldDescriptor\* f2) {/bool operator ()(const FieldDescriptor\* f1, const FieldDescriptor\* f2) const {/' \
"$1/src/google/protobuf/compiler/java/java_file.cc"
