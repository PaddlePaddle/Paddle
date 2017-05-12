# Name  
ls - list directory(ies)'s contents or file(s)'s attributes

# Synopsis
`ls [-r] <PFSPath> ...`

# Description

```
The following options are available:
       
-r
   List directory(ies) recursively
```

# Examples
- The following command lists a single file

```
paddle pfs ls  /pfs/$DATACENTER/home/$USER/text1.txt
```

- The following command lists directory contents

```
paddle pfs ls  / /pfs/$DATACENTER/home/$USER/folder
```
