# MIT License, see LICENSE
# Copyright (c) 2017 TensorPort
# Author: Malo Marrec, malo@goodailab.com

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

#tport 
from tensorport import TensorportClient as tport

from models.model import *
from utils.data_reader import *
from utils.view_steering_model import *

#Create logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """ Main wrapper"""

    # tport snippet 1 - get environment variables
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

    #Path to your data locally. This will enable to run the model both locally and on 
    # tensorport without changes
    PATH_TO_LOCAL_LOGS = os.path.expanduser('~/Documents/tensorport-self-driving-demo/logs/')
    ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser('~/Documents/comma/')
    #end of tport snippet 1


    #Flags
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # tport snippet 2: flags. 

    #Define the path from the root data directory to your data.
    #We use glob to match any .h5 datasets in Documents/comma locally, or in data/ on tensorport
    flags.DEFINE_string(
        "train_data_dir",
        tport().get_data_path(root=ROOT_PATH_TO_LOCAL_DATA,path='*/camera/training/*.h5'),
        """Path to training dataset. It is recommended to use get_data_path() 
        to define your data directory. If you set your dataset directory manually make sure to use /data/ 
        as root path when running on TensorPort cloud."""
        )
    flags.DEFINE_string(
        "val_data_dir",
        tport().get_data_path(root=ROOT_PATH_TO_LOCAL_DATA,path='*/camera/validation'),
        "Path to validation dataset."
        )

    flags.DEFINE_string("logs_dir",
        tport().get_logs_path(root=PATH_TO_LOCAL_LOGS),
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

    # end of tport snippet 2

    # Training flags - feel free to play with that!
    flags.DEFINE_integer("batch",256,"Batch size")
    flags.DEFINE_integer("time",1,"Number of frames per sample")
    flags.DEFINE_integer("steps_per_epoch",10,"Number of training steps per epoch")
    flags.DEFINE_integer("nb_val_batches",50,"Number of batches to get the average validation loss")
    flags.DEFINE_integer("nb_epochs",20,"Number of epochs")


    # Model flags - feel free to play with that!
    flags.DEFINE_float("dropout_rate1",.2,"Dropout rate on first dropout layer")
    flags.DEFINE_float("dropout_rate2",.5,"Dropout rate on second dropout layer")
    flags.DEFINE_float("starter_lr",1e-3,"Starter learning rate. Exponential decay is applied")
    flags.DEFINE_integer("fc_dim",512,"Size of the dense layer")
    flags.DEFINE_boolean("nogood",False,"Ignore `goods` filters.")


    # tport snippet 3: configure distributed environment
    def device_and_target():
        # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
        # Don't set a device.
        if FLAGS.job_name is None:
            print("Running single-machine training")
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
    # end of tport snippet 3


    if FLAGS.logs_dir is None or FLAGS.logs_dir == "":
        raise ValueError("Must specify an explicit `logs_dir`")
    if FLAGS.train_data_dir is None or FLAGS.train_data_dir == "":
        raise ValueError("Must specify an explicit `train_data_dir`")
    if FLAGS.val_data_dir is None or FLAGS.val_data_dir == "":
        raise ValueError("Must specify an explicit `val_data_dir`")

    TIME_LEN = 1 #1 video frame. Other not supported.

    # Define graph
    with tf.device(device):
        X = tf.placeholder(tf.float32, [FLAGS.batch, 3, 160, 320], name="X")
        Y = tf.placeholder(tf.float32,[FLAGS.batch,1], name="Y") # angle only
        S = tf.placeholder(tf.float32,[FLAGS.batch,1], name="S") #speed

        predictions = get_model(X,FLAGS)
        steering_summary = tf.summary.image("green-is-predicted",render_steering_tf(X,Y,S,predictions)) # Adding numpy operation to graph. Adding image to summary
        loss = get_loss(predictions,Y)
        val_loss = get_loss(predictions,Y) #separate val loss and training loss to avoid writing wrong summaries
        mean_loss, update_op = tf.contrib.metrics.streaming_mean(val_loss) #defining loss on batch

        training_summary = tf.summary.scalar('Training_Loss', loss)#add to tboard
        validation_summary = tf.summary.scalar('Validation_Loss', mean_loss) #replace with streamed loss

        #Batch generators
        gen_train = gen(FLAGS.train_data_dir, time_len=FLAGS.time, batch_size=FLAGS.batch, ignore_goods=FLAGS.nogood)

        gen_val = gen(FLAGS.val_data_dir, time_len=FLAGS.time, batch_size=FLAGS.batch, ignore_goods=FLAGS.nogood)

        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.starter_lr, global_step,1000, 0.96, staircase=True)
        
        train_step = (
            tf.train.AdamOptimizer(learning_rate)
            .minimize(loss, global_step=global_step)
            )

    def run_train_epoch(target,gen_train,FLAGS,epoch_index):
        """Restores the last checkpoint and runs a training epoch
        Inputs:
            - target: device setter for distributed work
            - FLAGS: 
                - requires FLAGS.logs_dir from which the model will be restored. 
                Note that whatever most recent checkpoint from that directory will be used.
                - requires FLAGS.steps_per_epoch
            - gen_train: training data generator
            - epoch_index: index of current epoch
        """

        hooks=[tf.train.StopAtStepHook(num_steps=FLAGS.steps_per_epoch)] # Increment number of required training steps
        i = 1

        with tf.train.MonitoredTrainingSession(master=target,
        is_chief=(FLAGS.task_index == 0),
        checkpoint_dir=FLAGS.logs_dir,
        hooks = hooks) as sess:

            while not sess.should_stop():
                batch_train = gen_train.next()

                feed_dict = {X: batch_train[0],
                                Y: batch_train[1],
                                S: batch_train[2]
                                }



                variables = [loss, learning_rate, train_step]
                current_loss, lr, _ = sess.run(variables, feed_dict)

                print("Iteration %s - Batch loss: %s" % ((epoch_index)*FLAGS.steps_per_epoch + i,current_loss))
                i+=1

    #Left to the reader - that could be done in a continuous fashion on a separate worker / thread
    def run_val(gen_val,FLAGS):
        """Restores the last checkpoint and runs validation
        Inputs:
            - target: device setter for distributed work
            - FLAGS: 
                - requires FLAGS.logs_dir from which the model will be restored. 
                Note that whatever most recent checkpoint from that directory will be used.
            - gen_val: validation data generator
        """
        with tf.device("/job:localhost/replica:0/task:0/cpu:0"):
            sess=tf.Session()  
            saver = tf.train.import_meta_graph('logs/model.ckpt-81.meta')
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.logs_dir))
            sess.run(tf.local_variables_initializer())
            for i in range(FLAGS.nb_val_batches):
                batch_val = gen_val.next()

                feed_dict = {X: batch_val[0],
                            Y: batch_val[1],
                            S: batch_val[2]
                            }
                current_loss = sess.run(update_op, feed_dict)
                print("Validation batch %s" % i)
                if i == (FLAGS.nb_val_batches - 1): #end of val batch, update loss
                    current_loss = sess.run(mean_loss)
                    print("Validation loss: %s" % (current_loss+1))

    for e in range(FLAGS.nb_epochs):
        run_train_epoch(target, gen_train, FLAGS, e)
        if FLAGS.task_index == 0: #if is_chief. Running val only on master node.
            run_val(gen_train, FLAGS)


if __name__ == "__main__": #Generators don't exit cleanly for some reason (GeneratorExit)
    main()


        

    