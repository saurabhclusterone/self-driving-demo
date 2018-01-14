# Demo of a Self-Driving Car Steering Model for TensorPort

[![join us on slack](https://slackin-altdyjrdgq.now.sh/badge.svg)](https://slackin-altdyjrdgq.now.sh)

This is a basic self-steering car model implemented in TensorFlow. It is used as a demo project to get started with the [TensorPort](https://tensorport.com) deep learning computation platform.


Follow the [Getting Started guide](https://docs.tensorport.com/v1.0/docs/getting-started) for TensorPort and read the author's blog post about building this demo [here](https://medium.com/towards-data-science/what-i-learnt-building-a-simple-self-steering-car-in-tensorflow-c8d7cab6f6d) and [here](https://medium.com/@malomarrec/how-to-write-distributed-tensorflow-code-with-an-example-on-tensorport-70bf3306adcb).


## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [License](#license)

## Install

To run this project, you need:

- [Python](https://python.org) 2.7 or 3.5 (or higher).
- [Git](https://git-scm.com/) and [Git Large File Storage (LFS)](https://git-lfs.github.com/)
- The tensorflow and tensorport libraries with `pip install tensorflow tensorport`
- A TensorPort account. Sign up [here](https://tensorport.com) for free if you don't have an account yet.

### Create the dataset

The model uses the Comma dataset, download it [here](https://tensorport-public.s3.amazonaws.com/comma-train.zip). Unzip the downloaded file and `cd` into the directory of the data.

Create a new repository with `git init`.

Use Git LFS to deal with the large files in the dataset: `git lfs track *.h5`.

Add and commit the changes with `git add -A`. This can take a moment due to the data size. Then run `git commit -m "added dataset"`.

Log into your TensorPort account using `tport login`.

Create a new dataset with `tport create dataset`.

### Get the code ready

Clone this repository to your local machine.

In [main_tf.py](/main_tf.py), change the flags at the top of the script to their appropriate values:

- `TENSORPORT_USERNAME`: your TensorPort username. This should be something like `johndoe`, not your email address!
- `LOCAL_LOG_LOCATION`: a location where you want the log files to be stored when running TensorPort on your local machine. Example: `~/Documents/tensorport-self-driving-demo/logs/`
- `LOCAL_DATASET_LOCATION`: the location of the comma dataset you downloaded above. Example: `~/Documents/data/comma`

## Usage

You can run the model either on your local machine or online on the TensorPort platform. Running a model is simple with the `tport` command line tool that comes included with the TensorPort Python package. Running the model locally is recommended for testing before uploading the code to TensorPort.

### Run locally

`cd` into the repository folder, then run the model on your local machine with:

```shell
tport run --local --requirements requirements.txt --mode single-node
```

In this example, single-node execution is selected. To create a distributed job, use the `--mode distributed` option instead of `--mode single-node`. This will simulate a distributed environment on your local machine.

### Run on TensorPort

To run the model on TensorPort, you first need a TensorPort account. Log in with `tport login`.

Create a new project with `tport create project`.

Push the project code to TensorPort with `git push tensorport master`.

Then, navigate to your dataset folder and push the dataset to TensorPort as well using `git push tensorport master`. This transfer will probably take a little while since the dataset is pretty large.

When the upload is complete, run the model on TensorPort using:

```bash
tport create job --name first-job --project YOUR_USERNAME/YOUR_PROJECT_NAME \
--datasets YOUR_USERNAME/YOUR_DATASET_NAME --module main_tf \
--framework-version 1.0.0 --requirements requirements.txt \
--mode distributed
--worker-type p2.xlarge --worker-replicas 3 \
--ps-type c4.2xlarge --ps-replicas 1 \
--time-limit 1h --description "This is the first run"
```

You can monitor the execution of your job on TensorPort using `tport watch`.

Instead of running the model from the command line, you can also use TensorPort's graphical web interface [Matrix](https://tensorport.com/matrix).

For a more detailed guide on how to run this project on TensorPort, check out the [Getting Started guide](https://docs.tensorport.com/v1.0/docs/getting-started). To learn more about TensorPort, visit our [website](https://tensorport.com).

## License

[MIT](LICENSE) © Good AI Lab, Inc.

Comma dataset and [data_reader.py](utils/data_reader.py) by [comma.ai](https://github.com/commaai/research), licensed as [specified](LICENSE_COMMA).
