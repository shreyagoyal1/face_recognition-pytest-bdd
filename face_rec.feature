Feature: Authentication using face recognition

  Scenario: face recognition accuracy
    Given the classifier encodes the image
    And classifier encodes the image name  
    When test image is taken, the face is detected to compare with folder faces to correctly recognize the face with its label
    