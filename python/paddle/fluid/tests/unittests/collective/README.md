# Steps to add a unittest in this directory
### step 1. Add Unittest files
    Add a file like test_c_identity.py
### step 2. Edit the `testslist.csv` file
    Add an item like test_c_identity in testslist.csv
    and specify the properties for the new unit test
    the properties are the following:
* `name`: the test's name
* `os`: The supported operator system, ignoring case. If the test run in multiple operator systems, use ";" to split systems, for example, `apple;linux` means the test runs on both Apple and Linux. The supported values are `linux`,`win32` and `apple`. If the value is empty, this means the test runs on all opertaor systems.
* `arch`: the device's architecture. similar to `os`, multiple valuse ars splited by ";" and ignoring case. The supported architectures are `gpu`, `xpu`, `ASCEND`, `ASCEND_CL` and `rocm`.
* `timeout`: timeout of a unittest, whose unit is second. Blank means defalut.
* `run_type`: run_type of a unittest. Supported values are `NIGHTLY`, `EXCLUSIVE`, `CINN`, `DIST`, `GPUPS`, `INFER`, `EXCLUSIVE:NIGHTLY`, `DIST:NIGHTLY`，which are case-insensitive. 
* `launcher`: the test launcher.Supported values are test_runner.py, dist_test.sh and custom scripts' name. Blank means test_runner.py.
* `num_port`: the number of port used in a distributed unit test. Blank means automatically distributed port. 
* `run_serial`: whether in serial mode. the value can be 1 or 0.Default (empty) is 0. Blank means defalut.
* `ENVS`: required environments. multiple envirenmonts are splited by ";".
* `conditions`: extra required conditions for some tests. The value is a list of boolean expression in cmake programmer, splited with ";". For example, the value can be `WITH_DGC;NOT WITH_NCCL` or `WITH_NCCL;${NCCL_VERSION} VERSION_GREATER_EQUAL 2212`,The relationship between these expressions is a conjunction.

### step 3. Generate CmakeLists.txt
    Run the cmd:
```bash
        python3 ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py -f ${PADDLE_ROOT}/python/paddle/fluid/tests/unittests/collective/testslist.csv
```
    Then the cmd generates a file named CMakeLists.txt in the save directory with the testslist.csv.
* usgae:  
    The command accepts --files/-f or --dirpaths/-d options, both of which accepts multiple values.  
    Option -f accepts a list of testslist.csv.  
    Option -d accepts a list of directory path including files named testslist.csv.  
    Type `python3 ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py --help` for details.

* note:  
When commiting the codes, you should commit both the testslist.csv and the generated CMakeLists.txt. Once you pulled the repo, you don't need to run this command until you modify the testslists.csv file.
    
### step 4. Build and test
    Build paddle and run ctest for the new unit test
