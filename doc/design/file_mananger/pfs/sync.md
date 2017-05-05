# Name  
pfs_cp - copy files and directories

# Synopsis
` pfs_cp
<LocalPath> <PFSPath> or <PFSPath> <LocalPath> or <PFSPath> <PFSPath>`

# Description

```
	-h, --help 
		Display this help and exit
		
	--version
       Output version information and exit

	--only-show-errors (boolean) 
		Only errors and warnings are displayed. All other output is suppressed.

	--page-size (integer) 
		The number of results to return in each response to a list operation. The default value is 1000 (the maximum allowed). Using a lower value may help if an operation times out.
		
	--preserve--links
       Reserve links when copy files
       
    -R, -r, --recursive
       Copy directories recursively
```

# Examples
- The following command cp a single file to pfs

```
pfs_cp ./text1.txt pfs://mydir/text1.txt
```

Output

```
upload ./text1.txt to pfs://mydir/text1.txt
```

- The following command cp pfs file to a local file

```
pfs_cp pfs://mydir/text1.txt ./text1.txt
```

Output

```
download pfs://mydir/text1.txt to ./text1.txt
```
