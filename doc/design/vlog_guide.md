This is a follow up for switching logging to VLOG. I feel we need a VLOG level guide for the PaddlePaddle developers. Here is what I have in mind

| Level | Field         | Example                       |
|-------|---------------|-------------------------------|
| 0     | System        | Library Loading               |
| 1     | Integration   | ?                             |
| 2     | Component     | executor, backward, optimizer |
| 3     | Moderate unit | operator, descBind            |
| 4     | Small unit    | attribute, typeDef            |

Any discussions are welcomed.

Reference: http://rpg.ifi.uzh.ch/docs/glog.html.
