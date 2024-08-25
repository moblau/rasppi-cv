# Imports
import mediapipe as mp
from picamera2 import Picamera2
import time
import cv2

# Initialize the pi camera
pi_camera = Picamera2()
# Convert the color mode to RGB
config = pi_camera.create_preview_configuration(main={"format": "RGB888"})

pi_camera.configure(config)

# Start the pi camera and give it a second to set up
pi_camera.start()
time.sleep(1)

	
def draw_pose(image, landmarks):
	''' 
	TODO Task 1
	
	Code to this fucntion to draw circles on the landmarks and lines
	connecting the landmarks then return the image.
	
	Use the cv2.line and cv2.circle functions.

	landmarks is a collection of 33 dictionaries with the following keys
		x: float values in the interval of [0.0,1.0]
		y: float values in the interval of [0.0,1.0]
		z: float values in the interval of [0.0,1.0]
		visibility: float values in the interval of [0.0,1.0]
		
	References:
	https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
	https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
	'''

	# copy the image
	landmark_image = image.copy()
	
	# get the dimensions of the image
	height, width, _ = image.shape
	for id,landmark in enumerate(landmarks.landmark):
		x_mapped = int(landmark.x*width)
		y_mapped = int(landmark.y*height)
		cv2.circle(landmark_image, (x_mapped,y_mapped), 10, (0,0,255), 2)


	#left face
	cv2.line(landmark_image, ( int((landmarks.landmark[0].x)*width),int((landmarks.landmark[0].y)*height)  ), ( int((landmarks.landmark[1].x)*width),int((landmarks.landmark[1].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[1].x)*width),int((landmarks.landmark[1].y)*height)  ), ( int((landmarks.landmark[2].x)*width),int((landmarks.landmark[2].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[2].x)*width),int((landmarks.landmark[2].y)*height)  ), ( int((landmarks.landmark[3].x)*width),int((landmarks.landmark[3].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[3].x)*width),int((landmarks.landmark[3].y)*height)  ), ( int((landmarks.landmark[7].x)*width),int((landmarks.landmark[7].y)*height)  ), (255,0,0), 5  )

	#right face
	cv2.line(landmark_image, ( int((landmarks.landmark[0].x)*width),int((landmarks.landmark[0].y)*height)  ), ( int((landmarks.landmark[4].x)*width),int((landmarks.landmark[4].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[4].x)*width),int((landmarks.landmark[4].y)*height)  ), ( int((landmarks.landmark[5].x)*width),int((landmarks.landmark[5].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[5].x)*width),int((landmarks.landmark[5].y)*height)  ), ( int((landmarks.landmark[6].x)*width),int((landmarks.landmark[6].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[6].x)*width),int((landmarks.landmark[6].y)*height)  ), ( int((landmarks.landmark[8].x)*width),int((landmarks.landmark[8].y)*height)  ), (255,0,0), 5  )

	#mouth
	cv2.line(landmark_image, ( int((landmarks.landmark[9].x)*width),int((landmarks.landmark[9].y)*height)  ), ( int((landmarks.landmark[10].x)*width),int((landmarks.landmark[10].y)*height)  ), (255,0,0), 5  )

	#left arm
	cv2.line(landmark_image, ( int((landmarks.landmark[11].x)*width),int((landmarks.landmark[11].y)*height)  ), ( int((landmarks.landmark[13].x)*width),int((landmarks.landmark[13].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[13].x)*width),int((landmarks.landmark[13].y)*height)  ), ( int((landmarks.landmark[15].x)*width),int((landmarks.landmark[15].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[15].x)*width),int((landmarks.landmark[15].y)*height)  ), ( int((landmarks.landmark[17].x)*width),int((landmarks.landmark[17].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[17].x)*width),int((landmarks.landmark[17].y)*height)  ), ( int((landmarks.landmark[19].x)*width),int((landmarks.landmark[19].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[19].x)*width),int((landmarks.landmark[19].y)*height)  ), ( int((landmarks.landmark[15].x)*width),int((landmarks.landmark[15].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[15].x)*width),int((landmarks.landmark[15].y)*height)  ), ( int((landmarks.landmark[21].x)*width),int((landmarks.landmark[21].y)*height)  ), (255,0,0), 5  )

	#right arm
	cv2.line(landmark_image, ( int((landmarks.landmark[12].x)*width),int((landmarks.landmark[12].y)*height)  ), ( int((landmarks.landmark[14].x)*width),int((landmarks.landmark[14].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[14].x)*width),int((landmarks.landmark[14].y)*height)  ), ( int((landmarks.landmark[16].x)*width),int((landmarks.landmark[16].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[16].x)*width),int((landmarks.landmark[16].y)*height)  ), ( int((landmarks.landmark[18].x)*width),int((landmarks.landmark[18].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[18].x)*width),int((landmarks.landmark[18].y)*height)  ), ( int((landmarks.landmark[20].x)*width),int((landmarks.landmark[20].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[20].x)*width),int((landmarks.landmark[20].y)*height)  ), ( int((landmarks.landmark[16].x)*width),int((landmarks.landmark[16].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[16].x)*width),int((landmarks.landmark[16].y)*height)  ), ( int((landmarks.landmark[22].x)*width),int((landmarks.landmark[22].y)*height)  ), (255,0,0), 5  )

	#chest
	cv2.line(landmark_image, ( int((landmarks.landmark[12].x)*width),int((landmarks.landmark[12].y)*height)  ), ( int((landmarks.landmark[11].x)*width),int((landmarks.landmark[11].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[11].x)*width),int((landmarks.landmark[11].y)*height)  ), ( int((landmarks.landmark[23].x)*width),int((landmarks.landmark[23].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[23].x)*width),int((landmarks.landmark[23].y)*height)  ), ( int((landmarks.landmark[24].x)*width),int((landmarks.landmark[24].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[24].x)*width),int((landmarks.landmark[24].y)*height)  ), ( int((landmarks.landmark[12].x)*width),int((landmarks.landmark[12].y)*height)  ), (255,0,0), 5  )	

	#left leg
	cv2.line(landmark_image, ( int((landmarks.landmark[23].x)*width),int((landmarks.landmark[23].y)*height)  ), ( int((landmarks.landmark[25].x)*width),int((landmarks.landmark[25].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[25].x)*width),int((landmarks.landmark[25].y)*height)  ), ( int((landmarks.landmark[27].x)*width),int((landmarks.landmark[27].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[27].x)*width),int((landmarks.landmark[27].y)*height)  ), ( int((landmarks.landmark[29].x)*width),int((landmarks.landmark[29].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[29].x)*width),int((landmarks.landmark[29].y)*height)  ), ( int((landmarks.landmark[31].x)*width),int((landmarks.landmark[31].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[31].x)*width),int((landmarks.landmark[31].y)*height)  ), ( int((landmarks.landmark[27].x)*width),int((landmarks.landmark[27].y)*height)  ), (255,0,0), 5  )

	#left leg
	cv2.line(landmark_image, ( int((landmarks.landmark[24].x)*width),int((landmarks.landmark[24].y)*height)  ), ( int((landmarks.landmark[26].x)*width),int((landmarks.landmark[26].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[26].x)*width),int((landmarks.landmark[26].y)*height)  ), ( int((landmarks.landmark[28].x)*width),int((landmarks.landmark[28].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[28].x)*width),int((landmarks.landmark[28].y)*height)  ), ( int((landmarks.landmark[30].x)*width),int((landmarks.landmark[30].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[30].x)*width),int((landmarks.landmark[30].y)*height)  ), ( int((landmarks.landmark[32].x)*width),int((landmarks.landmark[32].y)*height)  ), (255,0,0), 5  )
	cv2.line(landmark_image, ( int((landmarks.landmark[32].x)*width),int((landmarks.landmark[32].y)*height)  ), ( int((landmarks.landmark[28].x)*width),int((landmarks.landmark[28].y)*height)  ), (255,0,0), 5  )

	return landmark_image

def main():
	''' 
	TODO Task 2
		modify this fucntion to take a photo uses the pi camera instead 
		of loading an image

	TODO Task 3
		modify this function further to loop and show a video
	'''

	vw = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter('output.mp4', vw, 10, (640,480))
	# Create a pose estimation model 
	mp_pose = mp.solutions.pose
	
	# start detecting the poses
	with mp_pose.Pose(
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5) as pose:

		while True:
			image = pi_camera.capture_array()

			results = pose.process(image)
			if results.pose_landmarks != None:
				result_image = draw_pose(image, results.pose_landmarks)
				# cv2.imwrite('output.png', result_image)
			else:
				result_image = image	
			print(image.shape)

			cv2.imshow("Video", result_image)
			out.write(result_image)
			if cv2.waitKey(1) == ord('q'):
				break

		out.release()	

if __name__ == "__main__":
	main()
	print('done')
