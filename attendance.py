import cv2
import pandas as pd
import numpy as np
import urllib.request
import os
from datetime import datetime

path = r'C:\Users\Administrator\Pictures\images'
url='http://192.168.123.72/cam-hi.jpg'
# Ensure the path exists
os.makedirs(path, exist_ok=True)

if 'Attendance.csv' in os.listdir(path):    #listdir(os.path.join(os.getcwd(),path)):
    os.remove("Attendance.csv")
else:
    df = pd.DataFrame(list())
    df.to_csv("Attendance.csv")

images=[]
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def markAttendance(name):
    with open("Attendance.csv", 'r') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


# Initialize face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract name from classNames list
        # You can modify this part to recognize specific individuals if needed
        name = "Unknown"
        if len(classNames) > 0:
            name = classNames[0]
             # Assuming you only have one known person

        # Mark attendance
        markAttendance(name)

        # Display name on the image
        cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Frame', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cv2.destroyAllWindows()
