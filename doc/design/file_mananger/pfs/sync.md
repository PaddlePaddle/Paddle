# Name  
sync - sync directories. Recursively copies new and updated files from the source directory to the destination

# Synopsis
` sync [OPTIONS]...
<LocalPath> <PFSPath> or <PFSPath> <LocalPath> or <PFSPath> <PFSPath>`

# Description

```
--preserve--links
   Reserve links when copy links
```

# Examples
- The following command sync locally directory to pfs

```
paddle pfs sync ./dir1 pfs://mydir1
```

Output

```
upload ./dir1/text1.txt to pfs://mydir1/text2.txt
upload ./dir1/text2.txt to pfs://mydir1/text2.txt
...
```

- The following command sync pfs directory to local

```
paddle pfs sync pfs://mydir1 .
```

Output

```
download pfs://mydir1/text1.txt to ./text1.txt
download pfs://mydir1/text2.txt to ./text2.txt
...
```
