import cv2

# our video
video = cv2.VideoCapture('./testing_files/video2.mp4')

# our pre-trained car and pedestrian classifiers
car_tracker = cv2.CascadeClassifier('./raw/car_detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('./raw/haarcascade_fullbody.xml')

# Run forever until car stops or crashes
while True:
    # Read the current frame
    (read_successful, frame) = video.read()

    # safe coding
    if read_successful:
        # convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # draw rectangles around the cars
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # draw rectangles around the pedestrians
    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # display the image with detected cars
    cv2.imshow("Self Driving Car", frame)

    # wait for key input
    key = cv2.waitKey(1)

    # stop if Q is pressed
    if key == 81 or key == 113:
        break

# release the VideoCapture object
video.release()