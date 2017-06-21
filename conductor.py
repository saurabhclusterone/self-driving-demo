import os
import subprocess
import sys
from time import sleep
# a = os.system("python server.py --batch 200 --port 5557")


# b = os.system("python server.py --batch 200 --validation --port 5556")


# train_server = Popen("server.py --batch 200 --port 5557")
# val_server = Popen("server.py --batch 200 --validation --port 5556")


# train_server = subprocess.Popen([sys.executable, "server.py --batch 200 --port 5557"])

train_server = subprocess.Popen(['python server.py', '--batch','200', '--port','5557'], shell=True)
val_server = subprocess.Popen(['python server.py', '--batch','200', '--validaion','--port','5556'], shell=True)

processes = [train_server,val_server]

#DO the training
#...
# done

sleep(10)


#kill all the mess

for p in processes:
	if p.poll() is None:
		p.terminate()
		p.wait() #Wait until the process terminates