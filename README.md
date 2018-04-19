# Demo of a Self-Driving Car Steering Model for Clusterone

<p align="center">
<img src="co_logo.png" alt="Clusterone" width="200">
<br>
<br>
<a href="https://slackin-altdyjrdgq.now.sh"><img src="https://slackin-altdyjrdgq.now.sh/badge.svg" alt="join us on slack"></a>
</p>

This is a basic self-steering car model implemented in TensorFlow. It is used as a demo project to get started with the [Clusterone](https://clusterone.com) deep learning computation platform.


Follow the [Getting Started guide](https://docs.clusterone.com/v1.0/docs/get-started) for Clusterone and read the author's blog post about building this demo [here](https://medium.com/towards-data-science/what-i-learnt-building-a-simple-self-steering-car-in-tensorflow-c8d7cab6f6d) and [here](https://medium.com/@malomarrec/how-to-write-distributed-tensorflow-code-with-an-example-on-tensorport-70bf3306adcb).


## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [License](#license)

## Install

To run this project, you need:

- [Python](https://python.org) 2.7 or 3.5 (or higher).
- [Git](https://git-scm.com/) and [Git Large File Storage (LFS)](https://git-lfs.github.com/)
- The TensorFlow Python library. Get it with `pip install tensorflow`
- The Clusterone Python library. Install it with `pip install clusterone`
- A Clusterone account. Sign up [here](https://clusterone.com) for free if you don't have an account yet.

### Create the dataset

The model uses the Comma dataset, download it [here](https://tensorport-public.s3.amazonaws.com/comma-train.zip). Unzip the downloaded file and `cd` into the directory of the data.

Create a new repository with `git init`.

Use Git LFS to deal with the large files in the dataset: `git lfs track *.h5`.

Add and commit the changes with `git add -A`. This can take a moment due to the data size. Then run `git commit -m "added dataset"`.

Log into your Clusterone account using `just login`.

Create a new dataset with `just create dataset comma`.

Now, link the dataset to your local Git repository: `just ln dataset -p comma`.

Finally, upload the dataset to Clusterone with `git push clusterone master`.

### Get the code ready

Clone this repository to your local machine.

In [main_tf.py](/main_tf.py), change the flags at the top of the script to their appropriate values:

- `CLUSTERONE_USERNAME`: your Clusterone username. This should be something like `johndoe`, not your email address!
- `LOCAL_LOG_LOCATION`: a location where you want the log files to be stored when running Clusterone on your local machine. Example: `~/Documents/self-driving-demo/logs/`
- `LOCAL_DATASET_LOCATION`: the location of the comma dataset you downloaded above. Example: `~/Documents/data/comma`

## Usage

Running a job on Clusterone is simple with the `just` command line tool that comes included with the Clusterone Python package.

### Run on ClusterOne

To run the model on Clusterone, you first need a Clusterone account. Log in with `just login`.

`cd` into the directory where you cloned this repository to and create a new project with `just init project self-driving`.

Push the project code to Clusterone with `git push clusterone master`.

When the upload is complete, create a job to run the model on Clusterone:

```bash
just create job distributed --name first-job --project self-driving \
--datasets comma --module main_tf --framework tensorflow-1.0.0 --time-limit 1h
```

Now the final step is to start the job:

```bash
just start job -p self-driving/first-job
```

You can monitor the execution of your job on Clusterone using `just get events`.

Instead of running the model from the command line, you can also use Clusterone's graphical web interface [Matrix](https://clusterone.com/matrix).

For a more detailed guide on how to run this project on Clusterone, check out the [Getting Started guide](https://docs.clusterone.com/docs/get-started). To learn more about Clusterone, visit our [website](https://clusterone.com).

## License

[MIT](LICENSE) © Clusterone Inc.

Comma dataset and [data_reader.py](utils/data_reader.py) by [comma.ai](https://github.com/commaai/research), licensed as [specified](LICENSE_COMMA).
