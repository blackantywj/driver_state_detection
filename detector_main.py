from emotion_infer import emotion_detect
from eyes_detector import eyes_detect
from head_detector import head_detect
#import threading
import requests
import socket
import time
import random
import cv2
import json
from collections import deque
import numpy as np
#import emotion_infer
#import eyes_detector
url = "http://192.168.43.24:8000/post_data/"
cap = cv2.VideoCapture(0)
img_path = "./capture.jpg"
#server_receive_head = socket.socket()
#server_receive_eyes = socket.socket()
#server_receive_emotion = socket.socket()

#ip_port_head = ("127.0.0.1", 10010)
#ip_port_eyes = ("127.0.0.1", 10020)
#ip_port_emotion = ("127.0.0.1", 10030)
#server_receive_head.bind(ip_port_head)
#server_receive_eyes.bind(ip_port_eyes)
#server_receive_emotion.bind(ip_port_emotion)
#server_receive_head.listen(5)
#server_receive_eyes.listen(5)
#server_receive_emotion.listen(5)
#conn_head, addr_head = server_receive_head.accept()
#global num

#global Qeyes
#conn_eyes, addr_eyes = server_receive_eyes.accept()
#conn_emotion, addr_emotion = server_receive_emotion.accept()
def main():
	num = 0
	Qeyes = 0.35
	eyes_queue = deque(maxlen=5)
#	time.sleep(1)
	#while True:
		#try:
#	emotion_detect.run()
#	eyes_detect.run()
#	img_path = "./capture.jpg"
	while True:
		ret, frame = cap.read()
		cv2.imwrite(img_path, frame)
		eye_data, eyes_flag = eyes_detect.eyes_detector(frame)
		#if num == 0:
		#	#print("start")
		#	Qeyes = eye_data
		#	num+=1
		#	continue
		#print(Qeyes)
		#if Qeyes == 0:
		#	continue
		eyes = 1 - (eye_data/Qeyes)
		eyes_queue.append(eyes)
		fatigue = np.mean(eyes_queue)
		print(fatigue)
		if eyes_flag == True:
			emotion_array = emotion_detect.emotion_infer(img_path)
			head_data = head_detect.main(frame)
			#print(emotion_array)
			print(head_data)
			risk = emotion_array[0] * 0.2755 + emotion_array[1] * 0.0671 + emotion_array[2] * 0.0671 + emotion_array[3] * 0.0671 +emotion_array[4] * 0.0671 +emotion_array[5] * 0.0671 + emotion_array[6] * 0.0229 + fatigue * 0.3661 * 3
			request_data_json = {"vehicle_id": random.randint(0,100), "tired": round(fatigue * 100, 2), "e_anger": round(emotion_array[0] * 100,2), 
				"e_disgust": round(emotion_array[1] * 100,2), "e_fear": round(emotion_array[2] * 100, 2), 
				"e_happy": round(emotion_array[3] * 100, 2), "e_sad": round(emotion_array[4] * 100, 2), "e_surprised": round(emotion_array[5] * 100, 2), "e_normal": round(emotion_array[6] * 100, 2), "head_pose_x": head_data[0], "head_pose_y": head_data[1], "head_pose_z": head_data[2], "risk":round(risk * 100, 2)}
			request_data_file = {'img':open('./capture_add.jpg','rb')}
			print(json.dumps(request_data_json))
#			r = requests.post(url, data=request_data_json, files=request_data_file)
#			print(r)
			#print(head_data)
		#num += 1
#		data = conn.recv(1024)
#		print(1)
#		new_eyes_data = eyes_detect.dataQue.pop()
		#new_emotion_data = emotion_detect.averDataQue.pop()
		#print(new_emotion_data)
		#new_eyes_data = eyes_detect.data
#		print(new_eyes_data)
#				headdata = conn_head.recv(1024)
				
#		eyesdata = conn_eyes.recv(1024)
#		emotiondata = conn_emotion.recv(1024)
#		emotion_array = emotiondata.decode()
#		print(eyesdata)
#		request_data_json = {"vehicle_id": random.randint(0,100), "tired": int(eyesdata), "e_anger": emotion_array[0], 
#				"e_disgust": emotion_array[1], "e_fear": emotion_array[2], 
#				"e_happy": emotion_array[3], "e_sad": emotion_array[4], "e_surprised": emotion_array[5], "e_normal": emotion_array[6], "head_pose_x": 0.2322, "head_pose_y": 0.2322, "head_pose_z": 0.2322, "risk": 0.222}
#		request_data_file = {'capture': open(imagePath, 'rb')}
		#url = "http://192.168.1.107:8000/post_data/"
#		r = requests.post(url, data=request_data_json, files=request_data_file)
		#print(data)
		#except BaseException as e:
		#	e.print()
		#continue
#emotion_detect.run()
#eyes_detect=eyes_detector()
#eyes_detect.run()
main()
#thread1 = threading.Thread(target=main,args=())
#thread1.start()
