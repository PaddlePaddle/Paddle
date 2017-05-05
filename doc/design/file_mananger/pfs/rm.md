# Name  
rm - remove files or directories

# Synopsis
```
rm [OPTION]... <PFSPath>...
```

# Description

```
	-r, -R, --recursive 
		remove directories and their contents recursively

	--page-size (integer) 
		The number of results to return in each response to a list operation. The default value is 1000 (the maximum allowed). Using a lower value may help if an operation times out	
```

# Examples
- The following command deletes a single files

```
paddle pfs rm pfs://mydirectory/test1.txt
```

Output

```
delete pfs://mydirectory/test1.txt
```


- The following command deletes a  directory recursively.

```
paddle pfs rm -r pfs://mydirectory
```

Output

```
delete pfs://mydirectory/1.txt
delete pfs://mydirectory/2.txt
...
```
