import cv2, time
import serial

arduino =serial.Serial('COM3', 9600)
time.sleep(2)
print("Connecting to Arduino ...")

cam = cv2.VideoCapture(1) # (0) chỉ số của camera
time.sleep(2)

def detecteFaceDNN(net, frame, conf_threshold=0.7): # ngưỡng tin cậy 
    height = frame.shape[0]
    width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400), (104, 117, 123), False, False) 
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
            centre_s_x = width // 2 
            centre_s_y = height // 2 
            face_centre_x = (x1 + x2) // 2
            face_centre_y = (y1 + y2) // 2
            face = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0),
                          int(round(height / 200)))
            #circle_s = cv2.circle(frame, (centre_s_x, centre_s_y) , 3, (0,255,0), 3)
            centre_face = cv2.circle(frame, (face_centre_x, face_centre_y), 3 ,(0, 255, 0), 3)
            centre_s = cv2.rectangle(frame, (centre_s_x - 5, centre_s_x - 5), 
                        (centre_s_x + 5, centre_s_x + 5), (0, 255, 0), int(round(height / 200)))
            
            
            if x1 < centre_s_x < x2 and \
                y1 < centre_s_y < y2:
                pass    # chấm giữa mặt trong hình vuông giữa thì không diều khiển
            else:
                pan_position = int(face_centre_x - centre_s_x) # thay đổi góc quay ngang
                tilt_position = int(face_centre_y - centre_s_y)  # thay đổi góc quay dọc
                
                arduino.write(b'X')
                arduino.write(str(pan_position).encode())
                arduino.write(b'Y')
                arduino.write(str(tilt_position).encode())
    return frame, boxes

modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while 1:
    ret, frameOriginal = cam.read()
    frame = cv2.resize(frameOriginal, (700, 700)) 
    
    outDNN, boxes = detecteFaceDNN(net, frame)
    m_frame = cv2.flip(frame, 0)
    cv2.imshow("Face Detection", m_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
