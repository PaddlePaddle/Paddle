# Name  
rm - remove files or directories

# Synopsis
`rm [OPTION]... PFS_DIRECTORY...`

# Description
Create the directory, if it does not already exits

```
	-r, -R, --recursive 
		remove directories and their contents recursively
		
	-v, --version
       output version information and exit

	--only-show-errors (boolean) 
		Only errors and warnings are displayed. All other output is suppressed.

	--page-size (integer) 
		The number of results to return in each response to a list operation. The default value is 1000 (the maximum allowed). Using a lower value may help if an operation times out.		
```

# Examples
- The following command delete a single file

```
pfs_rm pfs://mydirectory/test1.txt
```

Output

```
delete pfs://mydirectory/test1.txt
```


- The following command delete a  directory recursively.

```
pfs_rm -r pfs://mydirectory
```

Output

```
delete pfs://mydirectory/1.txt
delete pfs://mydirectory/2.txt
...
```
