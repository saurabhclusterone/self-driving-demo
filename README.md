# Demo of a Self-Driving Car Steering Model for ClusterOne

<p align="center">
<img src="c1_logo.png" alt="ClusterOne" width="200">
<br>
<br>
<a href="https://slackin-altdyjrdgq.now.sh"><img src="https://slackin-altdyjrdgq.now.sh/badge.svg" alt="join us on slack"></a>
</p>

This is a basic self-steering car model implemented in TensorFlow. It is used as a demo project to get started with the [ClusterOne](https://clusterone.com) deep learning computation platform.


Follow the [Getting Started guide](https://docs.clusterone.com/v1.0/docs/getting-started) for ClusterOne and read the author's blog post about building this demo [here](https://medium.com/towards-data-science/what-i-learnt-building-a-simple-self-steering-car-in-tensorflow-c8d7cab6f6d) and [here](https://medium.com/@malomarrec/how-to-write-distributed-tensorflow-code-with-an-example-on-tensorport-70bf3306adcb).


## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [License](#license)

## Install

To run this project, you need:

- [Python](https://python.org) 2.7 or 3.5 (or higher).
- [Git](https://git-scm.com/) and [Git Large File Storage (LFS)](https://git-lfs.github.com/)
- The TensorFlow Python library. Get it with `pip install tensorflow`
- The ClusterOne Python library. Install it with `pip install clusterone`
- A ClusterOne account. Sign up [here](https://clusterone.com) for free if you don't have an account yet.

### Create the dataset

The model uses the Comma dataset, download it [here](https://tensorport-public.s3.amazonaws.com/comma-train.zip). Unzip the downloaded file and `cd` into the directory of the data.

Create a new repository with `git init`.

Use Git LFS to deal with the large files in the dataset: `git lfs track *.h5`.

Add and commit the changes with `git add -A`. This can take a moment due to the data size. Then run `git commit -m "added dataset"`.

Log into your ClusterOne account using `just login`.

Create a new dataset with `just create dataset`.

### Get the code ready

Clone this repository to your local machine.

In [main_tf.py](/main_tf.py), change the flags at the top of the script to their appropriate values:

- `CLUSTERONE_USERNAME`: your ClusterOne username. This should be something like `johndoe`, not your email address!
- `LOCAL_LOG_LOCATION`: a location where you want the log files to be stored when running ClusterOne on your local machine. Example: `~/Documents/self-driving-demo/logs/`
- `LOCAL_DATASET_LOCATION`: the location of the comma dataset you downloaded above. Example: `~/Documents/data/comma`

## Usage

You can run the model either on your local machine or online on the ClusterOne platform. Running a model is simple with the `just` command line tool that comes included with the ClusterOne Python package. Running the model locally is recommended for testing before uploading the code to ClusterOne.

### Run locally

`cd` into the repository folder, then run the model on your local machine with:

```shell
just run --local --requirements requirements.txt --mode single-node
```

In this example, single-node execution is selected. To create a distributed job, use the `--mode distributed` option instead of `--mode single-node`. This will simulate a distributed environment on your local machine.

### Run on ClusterOne

To run the model on ClusterOne, you first need a ClusterOne account. Log in with `just login`.

Create a new project with `just create project`.

Push the project code to ClusterOne with `git push clusterone master`.

Then, navigate to your dataset folder and push the dataset to ClusterOne as well using `git push clusterone master`. This transfer will probably take a little while since the dataset is pretty large.

When the upload is complete, run the model on ClusterOne using:

```bash
$ just create job --name first-job --project YOUR_USERNAME/YOUR_PROJECT_NAME \
--datasets YOUR_USERNAME/YOUR_DATASET_NAME --module main_tf \
--framework-version 1.0.0 --requirements requirements.txt \
--mode distributed
--worker-type p2.xlarge --worker-replicas 3 \
--ps-type c4.2xlarge --ps-replicas 1 \
--time-limit 1h --description "This is the first run"
```

You can monitor the execution of your job on ClusterOne using `just watch`.

Instead of running the model from the command line, you can also use ClusterOne's graphical web interface [Matrix](https://clusterone.com/matrix).

For a more detailed guide on how to run this project on ClusterOne, check out the [Getting Started guide](https://docs.clusterone.com/v1.0/docs/getting-started). To learn more about ClusterOne, visit our [website](https://clusterone.com).

## License

[MIT](LICENSE) © ClusterOne Inc.

Comma dataset and [data_reader.py](utils/data_reader.py) by [comma.ai](https://github.com/commaai/research), licensed as [specified](LICENSE_COMMA).
