# Demo of a self-driving car steering model for Tensorport

This is a demo of a self-steering car model for the tensorport deep learning computation platform.


## Run locally

#TODO


## Run on tensorport

1) Clone this repo

2) From the repo, run:
```bash
$ tport create_project
Display name:
$ comma-demo
Description [Tensorport Project]: 
$Demo of the comma self steering car.
Project created successfully
Waiting for Gitlab Repository
Saving Tensorport Config File
Tensorport Remote is already added
Project comma-demo and Repository are ready
```

3) Your project has now been created on tensorport, we just need to push the code. Run:
``` bash
$ git push tensorport master
```

4) To save you the hurdle of downloading data, we already created a dataset on tensorport. See the doc <#TODO HERE> for detailed instruction on how to load the data.

5) We are now ready to start distributed training on tensorport. From the CLI (see <#TODO HERE> for the GUI version):
``` bash
$ tport train
Display name: 
$ first-comma
Description [Tensorport Job]:
$ Run the comma demo
Please select project you want to use or specify --project parameter
0 | comma-demo | comma-demo
1 | mnist
Please select project to use, or type 0 to use latest one: [0]: 
$ 0
Project selected: comma-demo
Please select commit to use in a job, or type 0 to use latest one:
0 | #1e8262c318dde1d328883bf34ee22cafdb9149d0 | add missing requirements
1 | #8177d69b0c12eb580b02e2bca3123a991114e4c6 | removing suprefluous pygame use
2 | #4ac1b9397ba2d198f69988155ec83937c77d3065 | Updating requirements
3 | #6e609185d091d0e1be29e95e5c367cb6ea9f08bf | clean code
4 | #b96a793f1687ffc6dfae96bd27e6b07f607288c5 | Clean code and repo
...
$ 0
Commit selected: #1e8262c318dde1d328883bf34ee22cafdb9149d0
Please specify python module name to run [main]: 
$ main_tf
Please specify python path []: 
$
List of available instance types:
......................................
# | Name | CPU | GPU | Memory(GiB)
0 | c4.2xlarge | 1 | 0 | 10 
Please select instance type: 
$ 0
How many workers? [1]: 
$ 3
How many PS replicas? [1]: 
$ 1
Please specify requirements file: [requirements.txt]: 
$ 
Do you want to add a dataset to the job? Y/n [n]: y
Please select dataset you want to use
0 | comma-public | Public comma repo for comma demo
$ 0
Please select commit to use for this dataset in a job, or type 0 to use latest one:
$ 0
Please specify the dataset mounting point: []: 
$ 
Do you want to add another dataset to the job? Y/n [n]: 
$ n
Sending Job Create Request
Job Created Successfully
Starting job
```

6) Your job is now running on tensorport! You can access the matrix and view the logs and tensorboard.

 


## Dataset
The dataset consists of videos clips of variable size recorded at 20 Hz
with a camera mounted on the windshield of an Acura ILX 2016. In parallel to the videos
we also recorded some measurements such as car's speed, acceleration,
steering angle, GPS coordinates, gyroscope angles.

The dataset folder structure is the following:
```bash
+-- dataset
|   +-- camera
|   |   +-- 2016-04-21--14-48-08
|   |   ...
|   +-- labels
|   |   +-- 2016-04-21--14-48-08
|   |   ...
```

All the files come in hdf5 format and are named with the time they were recorded.
The camera dataset has shape `number_frames x 3 x 160 x 320` and `uint8` type.

See the original comma.ai repo for details.


## Requirements

[tensorport-0.8.9]
[tensorflow-1.0](https://github.com/tensorflow/tensorflow)  


## Credits
Author: Malo Marrec, malo@goodailab.com, (c) GoodAILab as specified in LICENSE

Data and data_reader.py -  https://github.com/commaai/research - Licensed as specified in LICENSE_COMMA