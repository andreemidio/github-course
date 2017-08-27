#First script for analyse the video
#Script withoud technics of machine learnig or deep learnig
#Script to study and discussion
#Write By Andre Emidio
#email :andresjc2008@gmail.com
#teste with python logs and github


#import library OpenCV, called cv2
import cv2

#capture the video
video =  cv2.VideoCapture(0)

#files haarcascade with classification the data of human face
#classifier frontal face
classifierFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
#classifier eye
classifierEye =  cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')
#classifier eye with glasses
classifierEyeGlass = cv2.CascadeClassifier('cascades\\haarcascade_eye_tree_eyeglasses.xml')7
# classifier smile
classifierSmile = cv2.CascadeClassifier('cascades\\haarcascade_smile.xml')

while True:
	conectado, frame =  video.read()
	#print(conectado)
	#print(frame)

	frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faceDetected =  classifierFace.detectMultiScale(frameGray, minSize=(70,70))
	#print(faceDetected)

	#loop for to reconized the face
	for(x, y, l, a) in faceDetected:
		#create rectangle to face using color red
		cv2.rectangle(frame, (x, y) , (x + l, y + a),(0,0,255),2)
		#create the region on Eye detection
		# region = y(height) +  a(area recognized) and x(width) + l(with recognize on the program)
		regionEye = frame[y:y+a, x:x+l]
		#transform de image in gray scale
		regioGrayEye = cv2.cvtColor(regionEye,cv2.COLOR_BGR2GRAY)
		#call the haar cascade on detected the eye area in the face
		detectedEye = classifierEye.detectMultiScale(regioGrayEye, scaleFactor = 1.11, minNeighbors = 8)

		#detect region smile
		# region = y(height) +  a(area recognized) and x(width) + l(with recognize on the program)
		regionSmile =  frame[y:y+a, x:x+a]
		#transform in gray scale
		regionGraySmile = cv2.cvtColor(regionSmile,cv2.COLOR_BGR2GRAY)
		#detected smile call the haar cascade smile
		#minNeighbor is a amount rectangle in image,about this rectangle far reality, example, recognized on knees 
		detectedSmile = classifierSmile.detectMultiScale(regionGraySmile, scaleFactor = 2, minNeighbors=5)
		
		#create the region on EyeGlass detection
		# region = y(height) +  a(area recognized) and x(width) + l(with recognize on the program
		regionEyeGlass = frame[y:y+a, x:x+l]
		##transform in grayscale
		regionGrayEyeGlass = cv2.cvtColor(regionEyeGlass, cv2.COLOR_BGR2GRAY)
		#call the haar cascade detected eye with glass 
		detectedEyeGlass =  classifierEyeGlass.detectMultiScale(regionGrayEyeGlass, scaleFactor=2,minNeighbors=5)


		# loop to detected eye
		for(ox, oy, ol, oa) in detectedEye:
        	#print(ox, oy, ol , oa )
 			cv2.rectangle(regionEye, (ox,oy),(ox + ol, oy + oa), (0,255,0),2)

 		# loop to detected smile
		for(so, sy, sl, sa) in detectedSmile:
			cv2.rectangle(regionSmile,(so,sy),(so + sl, sy + sa), (0,255,255),2)

		# loop to detected eye with glass
		for(sgo, sgy, sgl, sga) in detectedEyeGlass:
			cv2.rectangle(regionGrayEyeGlass,(sgo, sgy),(sgo + sgl, sgy + sga),(255,0,255),2)


	#show the video, frame to frame		
	cv2.imshow('Video', frame)

	# if you want to exit, press letter "Q"
	if cv2.waitKey(1) == ord('q'):
		break


#release the video
video.release()

#destroy and liberate the RAM memory
cv2.destroyAllWindows()
