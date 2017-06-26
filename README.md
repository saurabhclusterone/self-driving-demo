# Demo of a self-driving car steering model for Tensorport




## Dataset
The dataset consists of 10 videos clips of variable size recorded at 20 Hz
with a camera mounted on the windshield of an Acura ILX 2016. In parallel to the videos
we also recorded some measurements such as car's speed, acceleration,
steering angle, GPS coordinates, gyroscope angles. See the full `log` list [here](Logs.md).
These measurements are transformed into a uniform 100 Hz time base.

The dataset folder structure is the following:
```bash
+-- dataset
|   +-- camera
|   |   +-- 2016-04-21--14-48-08
|   |   ...
|   +-- log
|   |   +-- 2016-04-21--14-48-08
|   |   ...
```

All the files come in hdf5 format and are named with the time they were recorded.
The camera dataset has shape `number_frames x 3 x 160 x 320` and `uint8` type.
One of the `log` hdf5-datasets is called `cam1_ptr` and addresses the alignment
between camera frames and the other measurements.


## Requirements

[tensorport-0.8.9]
[tensorflow-1.0](https://github.com/tensorflow/tensorflow)  


## Credits
Author: Malo Marrec, malo@goodailab.com, (c) GoodAILab as specified in LICENSE
Data and data_reader.py -  https://github.com/commaai/research - Licensed as specified in LICENSE COMMA