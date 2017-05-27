# RecordIO

## Write

```go
f, e := os.Create("a_file.recordio")
w := recordio.NewWriter(f)
w.Write([]byte("Hello"))
w.Write([]byte("World!"))
w.Close()
f.Close()
```

## Read

1. Load chunk index:

   ```go
   f, e := os.Open("a_file.recordio")
   idx, e := recordio.LoadIndex(f)
   fmt.Println("Total records: ", idx.Len())
   f.Close()
   ```

2. Create one or more scanner to read a range of records.  The
   following example reads the range
   [1, 3), i.e., the second and the third records:

   ```go
   f, e := os.Open("a_file.recordio")
   s := recrodio.NewScanner(f, idx, 1, 3)
   for s.Scan() {
      fmt.Println(string(s.Record()))
   }
   if s.Err() != nil {
      log.Fatalf("Something wrong with scanning: %v", e)
   }
   f.Close()
   ```
