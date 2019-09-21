import dlib                                                         
import cv2
import sys
import numpy as np
from imutils import face_utils
import imutils
from collections import deque
import threading
import socket
#client_send = socket.socket()
#ip_port = ("127.0.0.1", 10020)
#client_send.connect(ip_port)
class eyes_detector():
	def __init__(self):
		self.EAR_THRESH = 0.2
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
		self.cameraCapture = cv2.VideoCapture(0)
		self.blink = False
		self.data = [0] * 2
		self.dataQue = deque()
		self.eyesdata = 0
	def eye_aspect_ratio(self, eye):
		A = np.linalg.norm(eye[1] - eye[5])
		B = np.linalg.norm(eye[2] - eye[4])
		C = np.linalg.norm(eye[0] - eye[3])
		ear = (A + B) / (2.0 * C)
		return ear
	def eyes_detector(self):
		print("start!")
		#frame = cv2.imread("./capture.jpg")
		ret, frame = self.cameraCapture.read()
		#cv2.imwrite("./capture.jpg", frame)
		#cv2.imshow("1",frame)
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
		#cv2.imshow("Frame", frame)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
		rects = self.detector(gray, 0)
		if len(rects) == 1:
			rect = rects[0]
			shape = self.predictor(gray, rect)
			shape = face_utils.shape_to_np(shape) 
			left_eye = shape[lStart:lEnd]
			right_eye = shape[rStart:rEnd]
			left_ear = self.eye_aspect_ratio(left_eye)
			right_ear = self.eye_aspect_ratio(right_eye)
			(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
			(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] 
                # compute the convex hull for the left and right eye, then visualize 
                # each of the eyes
			left_eye_hull = cv2.convexHull(left_eye)
			right_eye_hull = cv2.convexHull(right_eye)
			cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
			#cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
#			cv2.imshow("frame", frame)
#			if left_ear < self.EAR_THRESH or right_ear < self.EAR_THRESH:
				#self.data.append(left_ear)
				#self.data.append(right_ear)
			self.blink = True
			self.eyesdata = (left_ear + right_ear)/2
				#print(self.dataQue.pop())
			#else:
			#	self.blink = False
				#self.data = [0, 0]
				#self.dataQue.append(self.data)
		#else:
		#	self.blink = False
			#self.data = [0] * 2
		else:
			self.blink = False
		cv2.imshow("frame", frame)
		return self.eyesdata, self.blink
			#self.dataQue.append(self.data)
#	def main(self):
		#client_send = socket.socket()
		#ip_port = ("127.0.0.1", 10020)
		#client_send.connect(ip_port)
#		while True:
#			self.eyes_detector()
			#if self.blink == True:
			#	client_send.sendall(bytes(str(self.dataQue.pop()), encoding="utf-8"))
			#print(self.dataQue.pop())
#	def run(self):
#		thread1 = threading.Thread(target=self.main, args=())
#		thread1.start()
eyes_detect = eyes_detector()
while True:
	a = eyes_detect.eyes_detector()
	print(a)
