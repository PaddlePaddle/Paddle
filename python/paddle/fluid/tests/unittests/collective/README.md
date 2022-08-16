# Steps to add a unittest in this directory
### step 1. Add Unittest files
    Add a file like test_c_identity.py
### step 2. Edit the `testslist.csv` file
    Add an item like test_c_identity in testslist.csv
    and specify the properties for the new unit test
    the properties are the following:  
* `name`: the test's name
* `os`: The supported operator system, ignoring case. If the test run in multiple operator systems, use ";" to split systems, forexample, `apple;linux` means the test runs on both Apple and Linux. The supported values are `linux`,`win32` and `apple`. If the value is empty, this means the test runs on all opertaor systems.
* `arch`: the device's archetecture. similar to `os`, multiple valuse ars splited by ";" and ignoring case. The supported arhchetectures are `gpu`, `xpu`, `npu` and `rocm`.
* `timeout`: timeout of a unittest, whose unit is second.
* `run_type`: run_type of a unittest. Supported values are `NIGHTLY`, `EXCLUSIVE`, `CINN`, `DIST`, `GPUPS`, `INFER`，which are case-insensitive. Multiple Values are splited by ":".
* `launcer`: the test launcher.Supported values are test_runner.py, dist_test.sh and custom scripts' name.
* `dist_ut_port`: the starting port used in a distributed unit test
* `run_serial`: whether in serial mode. the value can be 1 or 0.
* `ENVS`: required environments. multiple envirenmonts are splited by ";".
* `conditons`: extra required conditions for some tests. the value is a boolean expression in cmake programmer.


### step 3. Generate CmakeLists.txt
    Run the cmd:
```bash
        python3 ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py -f ${PADDLE_ROOT}/python/paddle/fluid/tests/unittests/collective/testslist.csv
```
    Then the cmd generates a file named CMakeLists.txt in the save directory with the testslist.csv

* note:  
When commiting the codes, you should commit both the testslist.csv and the generated CMakeLists.txt. Once you pulled the repo, you don't need to run this command untill you modify the testslists.csv file.
    
### step 4. Build and test
    Build paddle and run ctest for the new unit test
