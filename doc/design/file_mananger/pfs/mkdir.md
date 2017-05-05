# Name  
pdf_mkdir  - mkdir directory(ies)

# Synopsis
`pfs_mkdir [OPTION]... PFS_DIRECTORY...`

# Description
Create the pfs directory(ies), if they do not already exist.

```
	-h, --help 
		display this help and exit
		
	--version
       output version information and exit

	--only-show-errors (boolean) 
		Only errors and warnings are displayed. All other output is suppressed.

	--page-size (integer) 
		The number of results to return in each response to a list operation. The default value is 1000 (the maximum allowed). Using a lower value may help if an operation times out.
```

# Examples
```
pfs_mkdir pfs://mydir1
```

Output

```
make directory pfs://mydir1
```
