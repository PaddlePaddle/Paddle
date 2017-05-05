# Name  
mv - move (rename) files or directories


# Synopsis
If destination already exist, please rm it first.
```
mv [OPTION]...
<LocalPath> <PFSPath> or <PFSPath> <LocalPath> or <PFSPath> <PFSPath>
```

# Description
```	
	-h, --help 
		display this help and exit
		
	-v, --version
       output version information and exit

	--only-show-errors (boolean) 
		Only errors and warnings are displayed. All other output is suppressed.

	--page-size (integer) 
		The number of results to return in each response to a list operation. The default value is 1000 (the maximum allowed). Using a lower value may help if an operation times out.
```

# Examples
- The following command move a single file to pfs

```
mv ./text1.txt pfs://mydirectory/test1.txt
```

Output

```
move text1.txt pfs://mydirectory/test1.txt
```
