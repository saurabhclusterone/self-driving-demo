# MIT License, see LICENSE
# Copyright (c) 2017 ClusterOne Inc.
# Author: Malo Marrec, malo@clusterone.com

"""
Runs distributed training of a self-steering car model.
"""


import time
import os
import logging
import traceback
import json
import glob
import tensorflow as tf
import numpy as np
import h5py

#clusterone
from clusterone import get_data_path, get_logs_path

from models.model import *
from utils.data_reader import *
from utils.view_steering_model import *

#Create logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Path to your data locally.
# Datasets on Clusterone is mounted as /data/username/dataset-name and
# this code assumes datasets are included at job creation on Clusterone
PATH_TO_LOCAL_LOGS = os.path.expanduser("~/logs/sdc/")
ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser("~/data/comma/")

def main():
    """ Main wrapper"""

    # clusterone snippet 1 - get environment variables
    try:
        job_name = os.environ['JOB_NAME']
        task_index = os.environ['TASK_INDEX']
        ps_hosts = os.environ['PS_HOSTS']
        worker_hosts = os.environ['WORKER_HOSTS']
    except:
        job_name = None
        task_index = 0
        ps_hosts = None
        worker_hosts = None

    #end of clusterone snippet 1

    environment = os.environ.get('CLUSTERONE_CLOUD') or os.environ.get('TENSORPORT_CLOUD')
    environment = 'clusterone-cloud' if environment else "local"

    #Flags
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # clusterone snippet 2: flags.

    #Define the path from the root data directory to your data.
    #We use glob to match any .h5 datasets in Documents/comma locally, or in data/ on ClusterOne
    flags.DEFINE_string(
        "train_data_dir",
        "data/*/*/camera/training/*.h5" if environment=="clusterone-cloud" else os.path.join(ROOT_PATH_TO_LOCAL_DATA,"camera/training/*.h5"),
        """Path to training dataset. It is recommended to use get_data_path()
        to define your data directory. If you set your dataset directory manually make sure to use /data/
        as root path when running on TensorPort cloud.
        On tensrport, the data will be mounted in /data/user/clusterone_dataset_name,
        so you can acces `path` with  /data/user/clusterone_dataset_name/path
        """
        )
    flags.DEFINE_string("logs_dir",
        "/logs/" if environment=="clusterone-cloud" else PATH_TO_LOCAL_LOGS,
        "Path to store logs and checkpoints. It is recommended"
        "to use get_logs_path() to define your logs directory."
        "If you set your logs directory manually make sure"
        "to use /logs/ when running on TensorPort cloud.")
    # Define worker specific environment variables. Handled automatically.
    flags.DEFINE_string("job_name", job_name,
                        "job name: worker or ps")
    flags.DEFINE_integer("task_index", task_index,
                        "Worker task index, should be >= 0. task_index=0 is "
                        "the chief worker task the performs the variable "
                        "initialization")
    flags.DEFINE_string("ps_hosts", ps_hosts,
                        "Comma-separated list of hostname:port pairs")
    flags.DEFINE_string("worker_hosts", worker_hosts,
                        "Comma-separated list of hostname:port pairs")
    # end of clusterone snippet 2

    logger.info("Reading data from %s " % FLAGS.train_data_dir)
    logger.info("Writing logs to %s " % FLAGS.logs_dir)

    # Training flags - feel free to play with that!
    flags.DEFINE_integer("batch",256,"Batch size")
    flags.DEFINE_integer("time",1,"Number of frames per sample")
    flags.DEFINE_integer("steps_per_epoch",10000,"Number of training steps per epoch")
    flags.DEFINE_integer("nb_epochs",200,"Number of epochs")


    # Model flags - feel free to play with that!
    flags.DEFINE_float("dropout_rate1",.2,"Dropout rate on first dropout layer")
    flags.DEFINE_float("dropout_rate2",.5,"Dropout rate on second dropout layer")
    flags.DEFINE_float("starter_lr",1e-6,"Starter learning rate. Exponential decay is applied")
    flags.DEFINE_integer("fc_dim",512,"Size of the dense layer")
    flags.DEFINE_boolean("nogood",False,"Ignore `goods` filters.")


    # clusterone snippet 3: configure distributed environment
    def device_and_target():
        # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
        # Don't set a device.
        if FLAGS.job_name is None:
            logging.info("Running single-machine training")
            return (None, "")

        # Otherwise we're running distributed TensorFlow.
        print("Running distributed training")
        if FLAGS.task_index is None or FLAGS.task_index == "":
            raise ValueError("Must specify an explicit `task_index`")
        if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
            raise ValueError("Must specify an explicit `ps_hosts`")
        if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
            raise ValueError("Must specify an explicit `worker_hosts`")

        cluster_spec = tf.train.ClusterSpec({
                "ps": FLAGS.ps_hosts.split(","),
                "worker": FLAGS.worker_hosts.split(","),
        })
        server = tf.train.Server(
                cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        if FLAGS.job_name == "ps":
            server.join()

        worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
        # The device setter will automatically place Variables ops on separate
        # parameter servers (ps). The non-Variable ops will be placed on the workers.
        return (
                tf.train.replica_device_setter(
                        worker_device=worker_device,
                        cluster=cluster_spec),
                server.target,
        )

    device, target = device_and_target()
    # end of clusterone snippet 3


    if FLAGS.logs_dir is None or FLAGS.logs_dir == "":
        raise ValueError("Must specify an explicit `logs_dir`")
    if FLAGS.train_data_dir is None or FLAGS.train_data_dir == "":
        raise ValueError("Must specify an explicit `train_data_dir`")
    # if FLAGS.val_data_dir is None or FLAGS.val_data_dir == "":
    #     raise ValueError("Must specify an explicit `val_data_dir`")

    logging.info("Reading data from %s" % FLAGS.train_data_dir)
    logging.info("Writing outputs and logs to %s" % FLAGS.logs_dir)

    # Define graph
    with tf.device(device):
        # X = tf.placeholder(tf.float32, [FLAGS.batch, 3, 160, 320], name="X")
        # Y = tf.placeholder(tf.float32,[FLAGS.batch,1], name="Y") # angle only
        # S = tf.placeholder(tf.float32,[FLAGS.batch,1], name="S") #speed

    	reader = DataReader(FLAGS.train_data_dir)
    	x, y, s = reader.read_row_tf()
        x.set_shape((3, 160, 320))
        y.set_shape((1))
        s.set_shape((1))

        X, Y, S = tf.train.batch([x,y,s], batch_size = FLAGS.batch)
        predictions = get_model(X,FLAGS)
        steering_summary = tf.summary.image("green-is-predicted",render_steering_tf(X,Y,S,predictions)) # Adding numpy operation to graph. Adding image to summary
        loss = get_loss(predictions,Y)
        training_summary = tf.summary.scalar('Training_Loss', loss)#add to tboard

        #Batch generators
        global_step = tf.train.get_or_create_global_step()#tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.starter_lr, global_step,1000, 0.96, staircase=True)

        train_step = (
            tf.train.AdamOptimizer(learning_rate)
            .minimize(loss, global_step=global_step)
            )

    def run_train_epoch(target,FLAGS,epoch_index):
        """Restores the last checkpoint and runs a training epoch
        Inputs:
            - target: device setter for distributed work
            - FLAGS:
                - requires FLAGS.logs_dir from which the model will be restored.
                Note that whatever most recent checkpoint from that directory will be used.
                - requires FLAGS.steps_per_epoch
            - epoch_index: index of current epoch
        """

        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.steps_per_epoch*epoch_index)] # Increment number of required training steps
        i = 1

        with tf.train.MonitoredTrainingSession(master=target,
        is_chief=(FLAGS.task_index == 0),
        checkpoint_dir=FLAGS.logs_dir,
        hooks = hooks) as sess:

            while not sess.should_stop():
                variables = [loss, learning_rate, train_step]
                current_loss, lr, _ = sess.run(variables)

                print("Iteration %s - Batch loss: %s" % ((epoch_index)*FLAGS.steps_per_epoch + i,current_loss))
                i+=1

    for e in range(FLAGS.nb_epochs):
        run_train_epoch(target, FLAGS, e)



if __name__ == "__main__":
    main()
