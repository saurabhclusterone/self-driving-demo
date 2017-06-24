import numpy as np
import h5py
import time
import os
import logging
import traceback
import tensorflow as tf
import json

#tport 
from tensorport import TensorportClient as tport

#Create logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# From comma.ai

def concatenate(camera_names, time_len):
	logs_names = [x.replace('camera', 'log') for x in camera_names]

	angle = []  # steering angle of the car
	speed = []  # steering angle of the car
	hdf5_camera = []  # the camera hdf5 files need to continue open
	c5x = []
	filters = []
	lastidx = 0

	for cword, tword in zip(camera_names, logs_names):
		try:
			with h5py.File(tword, "r") as t5:
				c5 = h5py.File(cword, "r")
				hdf5_camera.append(c5)
				x = c5["X"]
				c5x.append((lastidx, lastidx+x.shape[0], x))

				speed_value = t5["speed"][:]
				steering_angle = t5["steering_angle"][:]
				idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
				angle.append(steering_angle[idxs])
				speed.append(speed_value[idxs])

				goods = np.abs(angle[-1]) <= 200

				filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
				lastidx += goods.shape[0]
				# check for mismatched length bug
				print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0]))
				if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
					raise Exception("bad shape")

		except IOError:
			import traceback
			traceback.print_exc()
			print "failed to open", tword

	angle = np.concatenate(angle, axis=0)
	speed = np.concatenate(speed, axis=0)
	filters = np.concatenate(filters, axis=0).ravel()
	print "training on %d/%d examples" % (filters.shape[0], angle.shape[0])
	return c5x, angle, speed, filters, hdf5_camera

first = True

def datagen(data_dir, time_len=1, batch_size=256, ignore_goods=False):
	"""
	Parameters:
	-----------
	leads : bool, should we use all x, y and speed radar leads? default is false, uses only x
	
	data_dir: path to the data directory
	"""
	global first
	assert time_len > 0


	filter_files = os.listdir(data_dir)
	filter_files = [data_dir + "/" + f for f in filter_files if f.endswith(".h5")] #sorint .h5 + full path
	
	filter_names = sorted(filter_files)

	logger.info("Loading {} hdf5 buckets.".format(len(filter_names)))

	c5x, angle, speed, filters, hdf5_camera = concatenate(filter_names, time_len=time_len)
	filters_set = set(filters)

	logger.info("camera files {}".format(len(c5x)))

	X_batch = np.zeros((batch_size, time_len, 3, 160, 320), dtype='uint8')
	angle_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
	speed_batch = np.zeros((batch_size, time_len, 1), dtype='float32')

	while True:
		try:
			t = time.time()

			count = 0
			start = time.time()
			while count < batch_size:
				if not ignore_goods:
					i = np.random.choice(filters)
					# check the time history for goods
					good = True
					for j in (i-time_len+1, i+1):
						if j not in filters_set:
							good = False
					if not good:
						continue

				else:
					i = np.random.randint(time_len+1, len(angle), 1)

				# GET X_BATCH
				# low quality loop
				for es, ee, x in c5x:
					if i >= es and i < ee:
						X_batch[count] = x[i-es-time_len+1:i-es+1]
						break

				angle_batch[count] = np.copy(angle[i-time_len+1:i+1])[:, None]
				speed_batch[count] = np.copy(speed[i-time_len+1:i+1])[:, None]

				count += 1

			# sanity check
			assert X_batch.shape == (batch_size, time_len, 3, 160, 320)

			# logging.debug("loading image took: {}s".format(time.time()-t))
			# print("%5.2f ms" % ((time.time()-start)*1000.0))

			if first:
				print "X", X_batch.shape
				print "angle", angle_batch.shape
				print "speed", speed_batch.shape
				first = False

			yield (X_batch, angle_batch, speed_batch)

		except KeyboardInterrupt:
			raise
		except:
			traceback.print_exc()
			pass


def gen(data_dir, time_len=1, batch_size=256, ignore_goods=False):
	"""" Wrapper for datagen, taking only steering angle"""
	for data_row in datagen(data_dir, time_len, batch_size, ignore_goods):
		X, Y, _ = data_row
		Y = Y[:, -1]
		if X.shape[1] == 1:  # no temporal context
			X = X[:, -1]
		yield X, Y

#My code

def get_model(X):
	#ch, row, col = 3, 160, 320
	#Add the lambda here. Is it a mean substraction?

	conv1 = tf.layers.conv2d(
		tf.reshape(X,[FLAGS.batch,3,160,320]),
		filters = 16,
		kernel_size = (8,8),
		strides=(4, 4),
		padding='same',
		kernel_initializer=tf.contrib.layers.xavier_initializer(),
		bias_initializer=tf.zeros_initializer(),
		kernel_regularizer=None,
		name = 'conv1',
		activation = tf.nn.elu
		)

	conv2 = tf.layers.conv2d(
		conv1 ,
		filters = 32,
		kernel_size = (5,5),
		strides=(2, 2),
		padding='same',
		kernel_initializer=tf.contrib.layers.xavier_initializer(),
		bias_initializer=tf.zeros_initializer(),
		kernel_regularizer=None,
		name = 'conv2',
		activation = tf.nn.elu
		)

	conv3 = tf.layers.conv2d(
		conv2,
		filters = 64,
		kernel_size = (5,5),
		strides=(2, 2),
		padding='same',
		kernel_initializer=tf.contrib.layers.xavier_initializer(),
		bias_initializer=tf.zeros_initializer(),
		kernel_regularizer=None,
		name = 'conv3',
		activation = None
		)

	f1 = tf.contrib.layers.flatten(conv3)
	d1 = tf.nn.dropout(f1, FLAGS.dropout_rate1)
	e1 = tf.nn.elu(d1)
	dense1 = tf.layers.dense(e1, units = FLAGS.fc_dim)
	d2 = tf.nn.dropout(dense1, FLAGS.dropout_rate2)
	e2 = tf.nn.elu(d2)	
	dense2 = tf.layers.dense(e2, units = 1)

	return dense2


def get_loss(predictions,labels):
	# loss = tf.nn.l2_loss(predictions-labels)
	loss = tf.reduce_mean(tf.square(predictions-labels))
	return loss




if __name__ == "__main__":

	# tport snippet 1
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
	PATH_TO_LOCAL_LOGS = os.path.expanduser('~/Documents/research/logs/')
	ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser('~/Documents/research')
	#end of tport snippet 1


	#Flags
	flags = tf.app.flags
	FLAGS = flags.FLAGS

	# tport snippet 2: flags. 

	#Define the path from the root data directory to your data.
	# Here the data directory are split into training and validation, so defining two flags
	flags.DEFINE_string(
		"train_data_dir",
		tport().get_data_path(root=ROOT_PATH_TO_LOCAL_DATA,path='comma-light2/camera/training'),
		"""Path to training dataset. It is recommended to use get_data_path() 
		to define your data directory. If you set your dataset directory manually make sure to use /data/ 
		as root path when running on TensorPort cloud."""
		)
	flags.DEFINE_string(
		"val_data_dir",
		tport().get_data_path(root=ROOT_PATH_TO_LOCAL_DATA,path='comma-light2/camera/validation'),
		"Path to validation dataset."
		)
	flags.DEFINE_string("logs_dir",
		tport().get_logs_path(root=PATH_TO_LOCAL_LOGS),
		"Path to store logs and checkpoints. It is recommended"
		"to use get_logs_path() to define your logs directory."
		"If you set your logs directory manually make sure"
		"to use /logs/ when running on TensorPort cloud.")
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

	# Model flags
	flags.DEFINE_integer("batch",256,"Batch size")
	flags.DEFINE_integer("time",1,"Number of frames per sample")
	flags.DEFINE_integer("nb_train_step",100000,"Number of training steps")
	# flags.DEFINE_integer("nb_epochs",2,"Number of epochs")
	# flags.DEFINE_integer("epoch_size",10,"Size of epochs")
	# flags.DEFINE_integer("nb_val_batches",1,"Number of validation batches")
	flags.DEFINE_float("dropout_rate1",.2,"Dropout rate on first dropout layer")
	flags.DEFINE_float("dropout_rate2",.5,"Dropout rate on second dropout layer")
	flags.DEFINE_float("starter_lr",1e-3,"Starter learning rate. Exponential decay is applied")
	flags.DEFINE_integer("fc_dim",512,"Size of the dense layer")
	flags.DEFINE_boolean("nogood",False,"Ignore `goods` filters.")

	
	# tport snippet 3: configure distributed
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

	# print(FLAGS.train_data_dir)

	
	with tf.device(device):
		X = tf.placeholder(tf.float32, [FLAGS.batch, 3, 160, 320])
		Y = tf.placeholder(tf.float32,[FLAGS.batch,1]) # angle only
		# global_step = tf.Variable(0, trainable=False)	

		predictions = get_model(X)
		loss = get_loss(predictions,Y)

		#Batch generators
		gen_train = gen(FLAGS.train_data_dir, time_len=FLAGS.time, batch_size=FLAGS.batch, ignore_goods=FLAGS.nogood)
		gen_val = gen(FLAGS.val_data_dir, time_len=FLAGS.time, batch_size=FLAGS.batch, ignore_goods=FLAGS.nogood)

		global_step = tf.contrib.framework.get_or_create_global_step()
		learning_rate = tf.train.exponential_decay(FLAGS.starter_lr, global_step,100000, 0.96, staircase=True)
		# optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
		# train_step = optimizer.minimize(loss)
		
		
		train_step = (
			tf.train.AdamOptimizer(learning_rate)
			.minimize(loss, global_step=global_step)
			)

	hooks=[tf.train.StopAtStepHook(last_step=FLAGS.nb_train_step)]

	with tf.train.MonitoredTrainingSession(master=target,
		is_chief=(FLAGS.task_index == 0),
		checkpoint_dir=FLAGS.logs_dir,
		hooks = hooks) as sess:

		while not sess.should_stop():
			batch_train = gen_train.next()

			feed_dict = {X: batch_train[0],
							Y: batch_train[1]}

			variables = [loss,train_step]
			current_loss, _ = sess.run(variables, feed_dict)
			print("Batch loss: %s", current_loss)
		