import cv2, time
import serial

"""arduino =serial.Serial('COM3', 9600)
time.sleep(2)
print("Connecting to Arduino ...")"""

cam = cv2.VideoCapture(0)
time.sleep(2)


def detecteFaceDNN(net, frame, conf_threshold=0.85): #do chinh xac nhan dien khuon mat
    height = frame.shape[0]
    width = frame.shape[1]
    square_size = 50
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123), False, False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0),
                          int(round(height / 150)), 0, 0)
            cv2.rectangle(frame, ((width - square_size) // 2, (height - square_size) // 2),
                           (((width - square_size) // 2) + square_size, ((height - square_size) // 2) + square_size), (0, 0, 255), 2)
            
    """for x1, y1, x2, y2 in boxes:
        center_x = width // 2
        center_y = height // 2   
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
        
        if center_x - square_size // 2 <= face_center_x <= center_x + square_size // 2 and \
           center_y - square_size // 2 <= face_center_y <= center_y + square_size // 2:
            pass    #khuon mat trong hinh vuong giua thi khong dieu khien
        else:
            pan_position = int(((face_center_x - center_x) / (square_size // 2)) * 90)  #thay doi goc quay ngang
            tilt_position = int(((face_center_y - center_y) / (square_size // 2)) * 90)  #thay doi goc quay doc
            
            arduino.write(b'X')
            arduino.write(str(pan_position).encode())
            arduino.write(b'Y')
            arduino.write(str(tilt_position).encode())"""
    return frame, boxes

modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while 1:
    ret, frameOriginal = cam.read()
    if not ret:
        break
    frame = cv2.resize(frameOriginal, (600, 600))
    m_frame = cv2.flip(frame, 1)
    outDNN, boxes = detecteFaceDNN(net, frame)
    cv2.imshow("Face Detection", m_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
