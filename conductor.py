import os
import subprocess
import sys
from time import sleep
import threading
import os
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

from server import client_generator


#FLAGS. TOFDO USE TF FLAGS -> CLEAN
parser = argparse.ArgumentParser(description='Steering angle model trainer')
parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
parser.add_argument('--port', type=int, default=5557, help='Port of server.')
parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
parser.add_argument('--batch', type=int, default=64, help='Batch size.')
parser.add_argument('--epoch', type=int, default=1, help='Number of epochs.')
parser.add_argument('--epochsize', type=int, default=10, help='How many frames per epoch.')
parser.add_argument('--val_sample', type=int, default=10, help='Number of validation samples')
parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
parser.set_defaults(skipvalidate=False)
parser.set_defaults(loadweights=False)
args = parser.parse_args()





print("Starting data servers ...")

train_server = subprocess.Popen(['python server.py', '--batch','200', '--port',str(args.port)], shell=True)
val_server = subprocess.Popen(['python server.py', '--batch','200', '--validation','--port',str(args.val_port)], shell=True)


print(train_server)
print(val_server)


processes = [train_server,val_server]

print("...Started \n")

print("Starting training....")

def gen(hwm, host, port):
	for tup in client_generator(hwm=hwm, host=host, port=port):
		X, Y, _ = tup
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





model = get_model()
model.fit_generator(
	gen(20, args.host, port=args.port),
	samples_per_epoch=args.epochsize,
	nb_epoch=args.epoch,
	validation_data=None, #gen(20, args.host, port=args.val_port)
	nb_val_samples=0 #args.val_sample
)

print("...Ended")

print("Saving model weights and configuration file.")

if not os.path.exists("./outputs/steering_model"):
		os.makedirs("./outputs/steering_model")

model.save_weights("./outputs/steering_model/steering_angle.keras", True)
with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
	json.dump(model.to_json(), outfile)


#kill all the mess

for p in processes:
	if p.poll() is None:
		p.terminate()
		p.wait() #Wait until the process terminates