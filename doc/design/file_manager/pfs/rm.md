# Name  
rm - remove files or directories

# Synopsis
```
rm [-r] [-v] <PFSPath> ...
```

# Description

```
The following options are available:

-r 
	remove directories and their contents recursively
	
-v      
	Cause rm to be verbose, showing files after they are removed.
```

# Examples
- The following command deletes a single file:

```
paddle pfs rm /pfs/$DATACENTER/home/$USER/test1.txt
```

- The following command deletes a  directory recursively:

```
paddle pfs rm -r /pfs/$DATACENTER/home/$USER/folder
```
