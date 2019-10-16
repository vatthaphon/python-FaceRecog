import cv2
import face_recognition
import glob
import numpy as np
import os
import sys
import time

# from pathlib import Path
from PyQt5 import QtGui as pqtQtGui
from PyQt5 import QtWidgets as pqtQtWidgets
from PyQt5 import QtCore as pqtQtCore

IMAGES_PATH = './images'  # put your reference images in here
CAMERA_DEVICE_ID = 0
MAX_DISTANCE = 0.6  # increase to make recognition less strict, decrease to make more strict


def AV_load_image_file(imgs_path_p):
	"""
	Load an image to a workable format.

	Params:
		imgs_path_p	: The absolute path of the image.
	"""

	image_l = face_recognition.load_image_file(imgs_path_p)

	return image_l

def AV_isDirExist(abs_dir_p, isCreate_p=True):
# This function checks if the directory (e.g. path_2_corrtime_l = path_2_dir_p + "/avgpowtime/") exists.

	try:
		if not os.path.exists(abs_dir_p):
			if isCreate_p:
				os.makedirs(abs_dir_p)
				return True
			else:
				return False
		else:
			return True
	except OSError:
		return True

def AV_get_face_landmarks(image_p):
	"""
	face_landmarks_list is now an array with the locations of each facial feature in each face.
	face_landmarks_list[0]['left_eye'] would be the location and outline of the first person's left eye.
	"""

	face_landmarks_list_l = face_recognition.face_landmarks(image_p)

	return face_landmarks_list_l


def AV_draw_face_rectangles(image_p, face_locations_p, color_p, width_p):
    """
    Params:
        image_p	: It is the image obtained from calling AV_load_image_file.	
        face_locations_p : AV_get_face_embeddings_from_image
    """

    for (top_l, right_l, bottom_l, left_l) in face_locations_p:
        start_point_l = (left_l, top_l) 
        end_point_l = (right_l, bottom_l)

        cv2.rectangle(image_p, start_point_l, end_point_l, color_p, width_p)


def AV_get_face_embeddings_from_image(image_p, convert_to_rgb_p=False):
    """
    Take a raw image and run both the face detection and face embedding model on it.

    Params:
    	image_p	: It is the image obtained from calling AV_load_image_file.
    """

    # Convert from BGR to RGB if needed
    if convert_to_rgb_p:
        image_p = image_p[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations_l = face_recognition.face_locations(image_p)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings_l = face_recognition.face_encodings(image_p, face_locations_l)

    return face_locations_l, face_encodings_l

def AV_getImgs_Webcam(N_imgs_p, imgs_path_p, title_p="Test", T_delay_p=1):
	"""
	This function will get images from the webcam. The number of the images is N_imgs_p. Each images are captured T_delay_p sec. apart.
	"""

	AV_isDirExist(abs_dir_p=imgs_path_p)

	# open a connection to the camera
	video_capture_l = cv2.VideoCapture(CAMERA_DEVICE_ID)

	i_imgs_l = 0
	while video_capture_l.isOpened() and (i_imgs_l < N_imgs_p):
	# read from the camera in a loop, frame by frame

		time.sleep(T_delay_p)

		i_imgs_l = i_imgs_l + 1

		print("Taking images: " + str(i_imgs_l) + "/" + str(N_imgs_p))

		# Grab a single frame of video
		ok_l, frame_l = video_capture_l.read()

		# Display the image
		cv2.imshow(title_p, frame_l)

		# Hit 'q' on the keyboard to stop the loop
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		FN_l = imgs_path_p + "/whole_" + title_p + str(i_imgs_l) + ".jpg"
		cv2.imwrite(FN_l, frame_l)     # save frame as JPEG file  
		    
	# release handle to the webcam
	video_capture_l.release()

	# close the window (buggy on a Mac btw)
	cv2.destroyAllWindows()

def AV_compare_two_faces(image1_p, image2_p, tolerance_p=0.4):
    """
    Params:
    tolerance_p	: The lower value, the harder to match. The default value from the package is 0.6.
    """

    # picture_of_me = face_recognition.load_image_file("me.jpg")
    image1_face_encoding_l = face_recognition.face_encodings(image1_p)[0]


    # unknown_picture = face_recognition.load_image_file("unknown.jpg")
    image2_face_encoding_l = face_recognition.face_encodings(image2_p)[0]

    # Now we can see the two face encodings are of the same person with `compare_faces`!
    results_l = face_recognition.compare_faces([image1_face_encoding_l], image2_face_encoding_l, tolerance=tolerance_p)

    return results_l[0]

def AV_cmp_FacesVsFace(listOfFaces_p, aFace_p, tolerance_p=0.4):
    """
    Params:
        listOfFaces_p : a List of faces, i.e. [face1, face2, face3, ..., faceN]. facei is obtained from face_recognition.face_encodings.
    """

    return face_recognition.compare_faces(listOfFaces_p, aFace_p, tolerance=tolerance_p)

########## Classes
class AV_Webcam(pqtQtWidgets.QWidget):
    """
    This class is just for openning the webcam and show on the screen.
    """

    def __init__(self, fps_p=1, camera_port_p=0, width_p=640, height_p=480):

        super(AV_Webcam, self).__init__()

        self.width = width_p
        self.height = height_p
        self.image = pqtQtGui.QImage()
        self.camera_port = camera_port_p
        self.camera = None
        self.timer = pqtQtCore.QBasicTimer()
        self.fps = fps_p
        
    def stop_streaming(self):
        self.camera = None
        self.timer.stop()

    def start_streaming(self):
        self.camera = cv2.VideoCapture(self.camera_port)
        
        # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('video size:', width, height)

        self.timer.start(1000/self.fps, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            print("AV_Webcam: Fatal timer error.")
            return

        if self.camera is not None:
            ok_l, frame_l = self.camera.read()

            if ok_l:
                self.showFrame(frame_l)

    def showFrame(self, image_data):

        self.image = self.get_qimage(image_data)

        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()                

    def get_qimage(self, image_p: np.ndarray):

        height_l, width_l, colors_l = image_p.shape

        bytesPerLine_l = 3*width_l

        QImage = pqtQtGui.QImage

        image_l = QImage(image_p.data, width_l, height_l, bytesPerLine_l, QImage.Format_RGB888)

        return image_l.rgbSwapped()

    def paintEvent(self, event):

        painter_l = pqtQtGui.QPainter(self)

        painter_l.drawImage(0, 0, self.image)



class AV_WebCamFaceDetectionWidget(AV_Webcam):

    # face_encodings = pqtQtCore.pyqtSignal(np.ndarray)
    face_encodings = pqtQtCore.pyqtSignal(list)

    def __init__(self, where2sendFace_p, fps_p=1, camera_port_p=0):

        super(AV_WebCamFaceDetectionWidget, self).__init__(fps_p=fps_p, camera_port_p=camera_port_p)
        
        self.face_encodings.connect(where2sendFace_p)        

        self.red = (0, 0, 255)
        self.width = 2
        self.min_size = (30, 30)
        
    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            print("RecordVideo: Fatal timer error.")
            return

        if self.camera is not None:
            ok_l, frame_l = self.camera.read()

            if ok_l:
                self.showFrame(frame_l)    

    def showFrame(self, image_data):

        ##### Get images from the webcam        
        face_locations_l, face_encodings_l = self.detect_faces(image_p=image_data)

        self.face_encodings.emit(face_encodings_l)

        ##### Display rectangles around faces
        AV_draw_face_rectangles(image_p=image_data, face_locations_p=face_locations_l, color_p=self.red, width_p=self.width)

        self.image = self.get_qimage(image_data)

        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def detect_faces(self, image_p: np.ndarray):

        face_locations_l, face_encodings_l = AV_get_face_embeddings_from_image(image_p=image_p, convert_to_rgb_p=False)
        
        return face_locations_l, face_encodings_l

########## Test
def test_AV_getImgs_Webcam():
	AV_getImgs_Webcam(N_imgs_p=10, imgs_path_p="/Volumes/Data/Work/FIRST/EEG/UI/iAwareUI/images", title_p="Tom", T_delay_p=1)

def test_AV_get_face_embeddings_from_image():

	image_l = AV_load_image_file("/Volumes/Data/Work/FIRST/EEG/UI/iAwareUI/images/whole_Test2.jpg")
	# image_l = AV_load_image_file("/Volumes/Data/Work/FIRST/EEG/UI/iAwareUI/example.jpg")


	face_locations_l, face_encodings_l = AV_get_face_embeddings_from_image(image_p=image_l, convert_to_rgb_p=False)

	print(np.shape(face_locations_l))
	print(np.shape(face_encodings_l))

def test_AV_get_face_landmarks():
	image_l = AV_load_image_file("/Volumes/Data/Work/FIRST/EEG/UI/iAwareUI/images/whole_Test2.jpg")

	face_landmarks_list_l = AV_get_face_landmarks(image_p=image_l)

	print(face_landmarks_list_l)

def test_AV_compare_two_faces():
	
	for i_l in range(9):
		image1_l = AV_load_image_file("/Volumes/Data/Work/FIRST/EEG/UI/iAwareUI/images/whole_KT" + str(i_l + 1) + ".jpg")
		for j_l in range(9):
			image2_l = AV_load_image_file("/Volumes/Data/Work/FIRST/EEG/UI/iAwareUI/images/whole_Tom" + str(j_l + 1) + ".jpg")

			print("Test:" + str(i_l + 1) + "_" + str(j_l + 1) + ":" + str(AV_compare_two_faces(image1_p=image1_l, image2_p=image2_l, tolerance_p=0.4)))

			# if not AV_compare_two_faces(image1_p=image1_l, image2_p=image2_l):

				# print(str(i_l + 1) + "_" + str(j_l + 1))

if __name__ == "__main__":
	# test_AV_getImgs_Webcam()

	# test_AV_get_face_embeddings_from_image()
	
	# test_AV_get_face_landmarks()

	test_AV_compare_two_faces()
