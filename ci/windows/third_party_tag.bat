
pushd third_party\gloo
git fetch --tags
popd

pushd third_party\protobuf
git fetch --tags
popd

pushd third_party\gtest
git fetch --tags
popd

pushd third_party\pocketfft
git fetch --tags
popd

pushd third_party\pybind
git fetch --tags
popd

pushd third_party\brpc
git fetch --tags
popd

pushd third_party\rocksdb
git fetch origin 6.19.fb
popd
