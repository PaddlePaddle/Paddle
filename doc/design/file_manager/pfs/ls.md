# Name  
ls - list directory contents or a file attributes

# Synopsis
` ls [OPTION]... <PFSPath>`

# Description

```       
-R, -r, --recursive
   Copy directories recursively
   
--page-size (integer) 
	The number of results to return in each response to a list operation. The default value is 1000 (the maximum allowed). Using a lower value may help if operation time out.
```

# Examples
- The following command lists a single file

```
paddle pfs ls /pfs/mydir/text1.txt
```

Output

```
2017-05-5 17:04:30 text1.txt
```

- The following command lists directory contents

```
paddle pfs ls  /pfs/mydir
```

Output

```
2017-05-5 17:04:30 text1.txt
2017-05-5 17:04:30 text2.txt
...
```
