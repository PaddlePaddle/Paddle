# Name  
sync - sync directories. Recursively copies new and updated files from the source directory to the destination.

# Synopsis
``` 
sync [--preserve--links] [-v] <LocalPath> <PFSPath> 
sync [--preserve--links] [-v] <PFSPath> <LocalPath> 
sync [--preserve--links] [-v] <PFSPath> <PFSPath>`
```

# Description

```
The following options are available:

--preserve--links
   Reserve links when copy links.
   
-v 
	Cause sync to be verbose, showing files after their's synchronization is complete.
```

# Examples
- The following command sync locally directory to pfs.

```
paddle pfs sync ./dir1 /pfs/$DATACENTER/home/$USER/mydir1
```

- The following command sync pfs directory to local.

```
paddle pfs sync /pfs/$DATACENTER/home/$USER/mydir1 .
```
