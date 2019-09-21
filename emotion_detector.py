import random

from collections import deque

from openvino.inference_engine import IEPlugin, IENetwork

import numpy as np

import threading

from PIL import Image

from PIL import ImageFile

import cv2

import dlib

import requests

import threading

import socket

#client_send = socket.socket()

#ip_port = ("127.0.0.1", 10030)

#client_send.connect(ip_port)

global timer

#cameraCapture = cv2.VideoCapture(0)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class emotionInfer():

	def __init__(self):

		self.model_xml_path = "1.xml"

		self.model_bin_path = "1.bin"

		self.model_landmark_path = "./shape_predictor_68_face_landmarks.dat"

		self.model_facedet_path = "./haarcascade_frontalface_alt2.xml"

		self.capPic = "capture.jpg"

		self.result = {}

		self.maxLen = 5

		self.averData = [0] * 7

		self.averDataQue = deque()

		self.plugin = IEPlugin(device='MYRIAD')

		self.net = IENetwork.from_ir(model=self.model_xml_path, weights=self.model_bin_path)

		self.exec_net = self.plugin.load(network=self.net)

		self.dataQueue = deque(maxlen=self.maxLen)

		assert len(self.net.inputs.keys()) == 1

		assert len (self.net.outputs) == 1

		self.input_blob = next(iter(self.net.inputs))

		# input_blob = 'input'

		self.out_blob = next(iter(self.net.outputs))

		# out_blob   = 'output/BiasAdd'

	def capturePic(self):

		success, frame = cameraCapture.read()

		img_path = self.capPic

		cv2.imwrite(img_path, frame)

		return img_path

	def infer_init(self):

		plugin = IEPlugin(device='MYRIAD')

		net = IENetwork.from_ir(model=self.model_xml_path, weights=self.model_bin_path)

		exec_net = plugin.load(network=net)

		assert len(self.net.inputs.keys()) == 1

		assert len (self.net.outputs) == 1

		input_blob = next(iter(net.inputs))

		# input_blob = 'input'

		out_blob = next(iter(net.outputs))

		# out_blob   = 'output/BiasAdd'

	def pre_process_image(self, img_path):

	# Model input format

		n, c, h, w = [1, 3, 224, 224]

		image = Image.open(img_path)

		processedImg = image.resize((h, w), resample=Image.BILINEAR)

    # Normalize to keep data between 0 - 1

		processedImg = (np.array(processedImg) - 0) / 255.0

    # Change data layout from HWC to CHW

		processedImg = processedImg.transpose((2, 0, 1))

		processingImg = processedImg.reshape((n, c, h, w))

		return image, processingImg, img_path

	def emotion_infer(self, cap):

		#print(1)

		#self.averData = [0] * 7

		image, processedImg, imagePath = self.pre_process_image(cap)

		infer_result = self.exec_net.infer(inputs={self.input_blob: processedImg})

		emotion_array = infer_result["dense_2/Softmax"][0]

		#request_data_json = {"vehicle_id": random.randint(0,100), "latitude": "12.34", 

                # "longitude": "12.34", "tired": "12.34", "生气": emotion_array[0], 

                # "厌恶": emotion_array[1], "恐惧": emotion_array[2], 

                # "开心": emotion_array[3], "伤心": emotion_array[4], "惊讶": emotion_array[5], "正常": emotion_array[6]}

		#self.result = request_data_json

		self.dataQueue.append(emotion_array)

		#print(len(self.dataQueue))

		self.averData = [0] * 7

		while len(self.dataQueue) == 5:

			for data in self.dataQueue:

				#print(data)

				for num in range(7):

					self.averData[num] += data[num]

			self.averData = [i/5 for i in self.averData]

			#print(self.averData)

			#self.averDataQue.appendleft(self.averData)

			break

		return self.averData

				#print(self.averData[num])

		#print(request_data_json)

		#request_data_file = {'img': open(imagePath, 'rb')}

		#url = "http://192.168.1.107:8000/post_data/"

		#r = requests.post(url, data=request_data_json, files=request_data_file)

		#return emotion_array

	#def face_detector(self, cap):

	#	det = dlib.get_frontal_face_detector()

	#	frame = cv2.imread(cap)

	#	grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#print(grey)

	#	rects = det(grey, 1)

#	def main(self):

#		while True:

		#self.infer_init()

			#img = self.capturePic()

#			self.emotion_infer(self.capPic)

#			self.averData = [i/5 for i in self.averData]

#			self.averDataQue.appendleft(self.averData)

#			client_send.sendall(bytes(str(self.averDataQue.pop()), encoding="utf-8"))

			#print(self.averDataQue.pop())

			#a = self.eye_detector(img)

			#print(a)

#if  "__name__" = __main__:

#	def run(self):

#		thread1 = threading.Thread(target=self.main, args=())

#		thread1.start()

emotion_detect = emotionInfer()

#emotion_detect.run()

