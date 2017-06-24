import numpy as np
import h5py
import time
import os
import logging
import traceback
import tensorflow as tf
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.callbacks import TensorBoard

#tport 
from tensorport import TensorportClient as tport


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Batch extraction

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


def get_model(time_len=1):
	ch, row, col = 3, 160, 320  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
						input_shape=(ch, row, col),
						output_shape=(ch, row, col)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	return model


if __name__ == "__main__":

	#tport
	PATH_TO_LOCAL_LOGS = os.path.expanduser('~/Documents/research/logs/')
	ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser('~/Documents/research')


	flags = tf.app.flags
	FLAGS = flags.FLAGS

	flags.DEFINE_integer("batch",256,"Batch size")
	flags.DEFINE_integer("time",1,"Number of frames per sample")
	flags.DEFINE_integer("nb_epochs",1,"Number of epochs")
	flags.DEFINE_integer("epoch_size",10000,"Size of epochs")
	flags.DEFINE_integer("nb_val_samples",100,"Number of validation samples")
	flags.DEFINE_boolean("prep",False,"Use images preprocessed by vision model") #not supported right now
	flags.DEFINE_boolean("leads",False,"Use x, y and speed radar lead info.") # probably not supported right now
	flags.DEFINE_boolean("nogood",False,"Ignore `goods` filters.")
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

	# Model
	model = get_model()

	#Setup generator for getting batched data
	gen_train = gen(FLAGS.train_data_dir, time_len=FLAGS.time, batch_size=FLAGS.batch, ignore_goods=FLAGS.nogood)
	gen_val = gen(FLAGS.val_data_dir, time_len=FLAGS.time, batch_size=FLAGS.batch, ignore_goods=FLAGS.nogood)

	# Setup tensorboard

	if not os.path.exists(FLAGS.logs_dir): #maybe we need to add that dir creation in get_logs_path
		os.makedirs(FLAGS.logs_dir)
	board = TensorBoard(log_dir=FLAGS.logs_dir, histogram_freq=0,  
          write_graph=True, write_images=True)

	# import pdb; pdb.set_trace()
	model.fit_generator(
		gen_train,
		samples_per_epoch=FLAGS.epoch_size,
		nb_epoch=FLAGS.nb_epochs,
		validation_data=gen_val, #gen(20, args.host, port=args.val_port)
		nb_val_samples=FLAGS.nb_val_samples, #args.val_sample
		callbacks = [board]
	)

	print("Saving model weights and configuration file.")

	if not os.path.exists("./outputs/steering_model"):#TODO change to LOGS path, with a clean tport compatible flag
		os.makedirs("./outputs/steering_model")

	model.save_weights("./outputs/steering_model/steering_angle.keras", True)
	with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)
