### Backgroud
The data loading and preprocessing for training and inference in fluid is not very convenient to use.

Some problems are listed here:
- complicated to compose multiple 'readers' 
- not easy to adapt new file format with custom data records
- no common preprocessing pipeline for image
- efficient in data reading and processing need to be optimized

### Key problems to solve
1. Support multiple type of data source
     - memory(np.array, list, tuple or any iteratable objects)
    - local files(text, zip, tar, sequencefile, recordio, and so on)
    - hdfs files(basic input formats supported by hadoop)
    - other type of data repository, eg: 'mysql', 'redis', 'bigtable' and so one

2. Inteligently deserialize records from data source
    - automatically infer the schema of records
    - construct typed records from binary data stream using infered schema

3. Support any transformation defined by user to the records
    - basic transformations, eg: map, shuffle, buffer, batch, etc
    - frequently used transformations, eg: image crop, flip, disturbance, etc

4. Optimize the process of data loading and transformation
     - cache data locally(in memory or disk) when possible and permitted by user
     - optimize the data transformation, eg: rewrite it in C/C++, more concurrency, other advanced techniques like [weld](https://www.weld.rs/) 

5. Distributed data loading when train mode in multiple nodes
 
### How it works

### Project structure

### Usage
