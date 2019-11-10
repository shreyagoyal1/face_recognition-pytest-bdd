import pytest_bdd
import pytest
import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep 

#@pytest.fixture
#def path():
 #  #return encoded
  #  data=os.walk("./faces")
   # return(data)
    
# Scenarios
@pytest_bdd.scenario('face_rec.feature','face recognition accuracy', features_base_dir='', strict_gherkin=False)
def test_face_recognition():
    pass
    

# Given Steps
@pytest_bdd.given('the classifier encodes the image')
def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".jpeg"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


@pytest.fixture
def image():
    return image




# Given Steps
@pytest_bdd.given('classifier encodes the image name')
def unknown_image_encoded(image):
    """
    encode a face given the file name
    """
    from face_recognition.face_recognition_cli import image_files_in_folder
    my_dir = 'C:/Users/Shreya Goyal/Desktop/face recognition pytest bdd/faces/' 
    encoding = [] # Create an empty list for saving encoded files
    for i in os.listdir(my_dir): # Loop over the folder to list individual files
        image = my_dir + i
        image = face_recognition.load_image_file(image) # Run your load command
        image_encoding = face_recognition.face_encodings(image) # Run your encoding command
        encoding.append(image_encoding[0]) # Append the results to encoding_for_file list
        return encoding
 #   folder = os.walk("./faces")
 #   face = fr.load_image_file("faces/", img)
#   encoding = fr.face_encodings(face)[0]

 #   return encoding


@pytest.fixture
def im():
    return im



# When Steps
@pytest_bdd.when('test image is taken, the face is detected to compare with folder faces to correctly recognize the face with its label')
def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    image1 = cv2.imread("Shreya.jpeg")
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(image1)
    unknown_face_encodings = face_recognition.face_encodings(image1, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
#        assert name == "priyanka.jpg".split(".")[0]
        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(image1, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(image1, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image1, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)
        assert name == "Shreya.jpeg".split(".")[0]    

    # Display the resulting image
    while True:

        cv2.imshow('Video', image1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 

    
    print(classify_face("Shreya.jpeg"))





