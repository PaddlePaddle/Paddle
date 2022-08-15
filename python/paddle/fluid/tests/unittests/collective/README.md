# Steps to add a unittests in this directory
### step 1. Add Unittests files
    Add a file like test_c_identity.py
### step 2. Edit the `testslist.csv` file
    Add an item like test_c_identity in testslist.csv
    and specify the properties for the new unit test
### step 3. Generate CmakeLists.txt
    Run the cmd:
```bash
        python3 ${PADDLE_ROOT}/tools/gen_ut_cmakelists.py -f ${PADDLE_ROOT}/python/paddle/fluid/tests/unittests/collective/testslist.csv
```
    Then the cmd generates a file named CMakeLists.txt in the save directory with the testslist.csv

* note:  
When commiting the codes, you should commit both the testslist.csv and the generated CMakeLists.txt. Once you pulled the repo, you don't need to run this command untill you modify the testslists.csv file.
    
### step 4. Run cmake and make and ctest
    Build paddle and run ctest for the new unit test
