import os
import sys
import re
import json


def get_all_paddle_file(rootPath):
    """get all file in Paddle repo: paddle/fluild, python"""
    traverse_files = ['%s/paddle/fluid' %rootPath, '%s/python' %rootPath]
    all_file_paddle = '%s/build/all_file_paddle' %rootPath
    all_file_paddle_list = []
    with open(all_file_paddle, 'w') as f:
        for filename in traverse_files:
            g = os.walk(filename)
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    all_file_paddle_list.append(os.path.join(path, file_name))
    return all_file_paddle_list

def get_all_uts(rootPath):
    all_uts_paddle = '%s/build/all_uts_paddle' %rootPath
    os.system('cd %s/build && ctest -N -V | grep -Ei "Test[ \t]+#" | grep -oEi "\w+$" > %s' %(rootPath, all_uts_paddle))
 
def remove_useless_file(rootPath):
    """remove useless file in ut_file_map.json"""
    all_file_paddle_list = get_all_paddle_file(rootPath)
    ut_file_map_new = {}
    ut_file_map = "%s/build/ut_file_map.json" %rootPath
    with open(ut_file_map, 'r') as load_f:    
        load_dict = json.load(load_f)
    for key in load_dict: 
        print(key, len(load_dict[key]))  
        if key in all_file_paddle_list:
            ut_file_map_new[key] = load_dict[key]
        else:
            print('notxxx:::: %s' %key)
            print(key, len(load_dict[key]))  
    
    with open("%s/build/ut_file_map.json" %rootPath, "w") as f:
        json.dump(ut_file_map_new, f, indent=4)   
        print("ut_file_map success!!")

def handle_ut_file_map(rootPath):
    utNotSuccess = ''
    ut_map_path = "%s/build/ut_map" %rootPath
    files= os.listdir(ut_map_path)
    ut_file_map = {}
    for ut in files:
        print("ut: %s" %ut)
        coverage_info = '%s/%s/coverage.info.tmp' %(ut_map_path, ut)
        if os.path.exists(coverage_info): 
            filename = '%s/%s/%s.txt' %(ut_map_path, ut, ut)
            f = open(filename)
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').strip()
                if line == '':
                    continue
                elif line.startswith('/paddle/build'):
                    source_file = line.replace('/build', '')
                    #source_file = re.sub('.pb.*', '.proto', source_file)
                elif 'precise test map fileeee:' in line:
                    source_file = line.split('precise test map fileeee:')[1].strip()
                else:
                    source_file = line 
                print(source_file)
                if source_file not in ut_file_map:
                    ut_file_map[source_file] = []
                if ut not in ut_file_map[source_file]:
                    ut_file_map[source_file].append(ut)
                
        else:
            utNotSuccess += '^%s$|' %utNotSuccess
    print(utNotSuccess)

    with open("%s/build/utNotSuccess.log" %rootPath, "w") as f:
        
        json.dump(ut_file_map, f, indent=4)   
        print("ut_file_map success!!")  

    with open("%s/build/ut_file_map.json" %rootPath, "w") as f:
        json.dump(ut_file_map, f, indent=4)   
        print("ut_file_map success!!")
    


    
    '''
    print(len(ut_file_map))
    for key in ut_file_map: 
        print(key, len(ut_file_map[key]))  
    '''
    with open("/paddle/build/ut_file_map.json",'r') as load_f:    
        load_dict = json.load(load_f)
    #print(len(load_dict))
    
    for key in load_dict: 
        print(key, len(load_dict[key]))  

def notsuccessfuc(rootPath):
    utNotSuccess = ''
    ut_map_path = "%s/build/ut_map" %rootPath
    files= os.listdir(ut_map_path) 
    ut_file_map = {}
    count = 0
    # ut failed!!
    for ut in files:
        coverage_info = '%s/%s/coverage.info.tmp' %(ut_map_path, ut)
        if os.path.exists(coverage_info):    
            pass
        else:
            count = count +1
            utNotSuccess = utNotSuccess + '^%s$|' %ut
    # ut not exec
    get_all_uts(rootPath)
    with open("/paddle/build/all_uts_paddle", "r") as f:
        data = f.readlines()
    for ut in data:
        ut = ut.replace('\n', '').strip()
        if ut not in files:
            count = count +1
            utNotSuccess = utNotSuccess + '^%s$|' %ut
    print("utNotSuccess count: %s" %count)
    if utNotSuccess != ''
        os.system('echo %s > %s/build/utNotSuccess' %(utNotSuccess[:-1], rootPath))
    
if __name__ == "__main__":
    func = sys.argv[1]
    if func == 'get_not_success_ut':
        rootPath = sys.argv[2]
        notsuccessfuc(rootPath)
