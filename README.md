# Demo of a Self-Driving Car Steering Model for Clusterone

<p align="center">
<img src="co_logo.png" alt="Clusterone" width="200">
<br>
<br>
<a href="https://slackin-altdyjrdgq.now.sh"><img src="https://slackin-altdyjrdgq.now.sh/badge.svg" alt="join us on slack"></a>
</p>

This is a basic self-steering car model implemented in TensorFlow. It is used as a demo project to get started with the [Clusterone](https://clusterone.com) deep learning computation platform.


Follow the [Getting Started guide](https://docs.clusterone.com/docs/get-started) for Clusterone and read the author's blog post about building this demo [here](https://medium.com/towards-data-science/what-i-learnt-building-a-simple-self-steering-car-in-tensorflow-c8d7cab6f6d) and [here](https://medium.com/@malomarrec/how-to-write-distributed-tensorflow-code-with-an-example-on-tensorport-70bf3306adcb).


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

## Usage

Running a job on Clusterone is simple with the `just` command line tool that comes included with the Clusterone Python package.

### Run on Clusterone

To run the model on Clusterone, you first need a Clusterone account. Log in with `just login`.

`cd` into the directory where you cloned this repository to and create a new project with `just init project self-driving-demo`. The data is already uploaded to Clusterone, so you don't need to worry about it.

Push the project code to Clusterone with `git push clusterone master`.

When the upload is complete, create a job to run the model on Clusterone:

```bash
just create job distributed --name first-job --project self-driving-demo \
--datasets tensorbot/self-driving-demo-data --module main_tf
```

Now the final step is to start the job:

```bash
just start job -p self-driving-demo/first-job
```

You can monitor the execution of your job on Clusterone using `just get events`.

Instead of running the model from the command line, you can also use Clusterone's graphical web interface [Matrix](https://clusterone.com/matrix).

For a more detailed guide on how to run this project on Clusterone, check out the [Getting Started guide](https://docs.clusterone.com/docs/get-started). To learn more about Clusterone, visit our [website](https://clusterone.com).

## License

[MIT](LICENSE) © Clusterone Inc.

Comma dataset and [data_reader.py](utils/data_reader.py) by [comma.ai](https://github.com/commaai/research), licensed as [specified](LICENSE_COMMA).
