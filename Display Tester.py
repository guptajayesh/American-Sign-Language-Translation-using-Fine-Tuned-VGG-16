
# coding: utf-8

# In[1]:


# organize imports
import cv2
import imutils
import numpy as np
import keras


# global variables
bg = None


# In[4]:


def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


# In[5]:


#-------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
#-------------------------------------------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


# In[4]:


from keras.models import load_model
classifier = load_model('my_model.h5')


# In[5]:


def predictor(test_image):
       import numpy as np
       from keras.preprocessing import image
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)


       
       if result[0][0] == np.amax(result[0]):
              return 'A'
       elif result[0][1] == np.amax(result[0]):
              return 'B'
       elif result[0][2] == np.amax(result[0]):
              return 'C'
       elif result[0][3] == np.amax(result[0]):
              return 'D'
       elif result[0][4] == np.amax(result[0]):
              return 'E'
       elif result[0][5] == np.amax(result[0]):
              return 'F'
       elif result[0][6] == np.amax(result[0]):
              return 'G'
       elif result[0][7] == np.amax(result[0]):
              return 'H'
       elif result[0][8] == np.amax(result[0]):
              return 'I'
       elif result[0][9] == np.amax(result[0]):
              return 'J'
       elif result[0][10] == np.amax(result[0]):
              return 'K'
       elif result[0][11] == np.amax(result[0]):
              return 'L'
       elif result[0][12] == np.amax(result[0]):
              return 'M'
       elif result[0][13] == np.amax(result[0]):
              return 'N'
       elif result[0][14] == np.amax(result[0]):
              return 'O'
       elif result[0][15] == np.amax(result[0]):
              return 'P'
       elif result[0][16] == np.amax(result[0]):
              return 'Q'
       elif result[0][17] == np.amax(result[0]):
              return 'R'
       elif result[0][18] == np.amax(result[0]):
              return 'S'
       elif result[0][19] == np.amax(result[0]):
              return 'T'
       elif result[0][20] == np.amax(result[0]):
              return 'U'
       elif result[0][21] == np.amax(result[0]):
              return 'V'
       elif result[0][22] == np.amax(result[0]):
              return 'W'
       elif result[0][23] == np.amax(result[0]):
              return 'X'
       elif result[0][24] == np.amax(result[0]):
              return 'Y'
       elif result[0][25] == np.amax(result[0]):
              return 'Z'


# In[6]:


#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 0, 0, 100, 100

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)


        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)
                resized_image = cv2.resize(thresholded, (50, 50))
                resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB)
                a = predictor(resized_image/255.0)
                print a
        
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()

