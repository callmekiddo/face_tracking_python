import cv2, time
import serial

arduino =serial.Serial('COM6', 9600)
#time.sleep(2)
print("Connecting to Arduino ...")

cam = cv2.VideoCapture(1) # (0) chỉ số của camera
#time.sleep(2)

def detecteFaceDNN(net, frame, conf_threshold=0.62): # ngưỡng tin cậy 
    height = frame.shape[0]
    width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (104, 117, 123), False, False) 
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    first_face = False
    square_size = 35
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * width) # góc trên bên trái 
            y1 = int(detections[0, 0, i, 4] * height) # góc dưới bên trái
            x2 = int(detections[0, 0, i, 5] * width) # góc trên bên phải
            y2 = int(detections[0, 0, i, 6] * height) # góc dưới bên phải
            
            
            boxes.append([x1, y1, x2, y2]) 
            centre_s_x = width // 2 
            centre_s_y = height // 2 
            face_centre_x = (x1 + x2) // 2
            face_centre_y = (y1 + y2) // 2
            if not first_face:
                first_face = True
                face = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0),
                            int(round(height / 200)))
                circle_s = cv2.circle(frame, (centre_s_x, centre_s_y) , 3, (0,255,0), 3)
            
            if(x1 <= centre_s_x <= x2 and y1 <= centre_s_y <= y2) :
                pass    # chấm giữa mặt trong hình vuông giữa thì không diều khiển
            else:
                
                pan_position = int(face_centre_x - centre_s_x)
                tilt_position = int(face_centre_y - centre_s_y) 
                arduino.write(b'X')
                arduino.write(str(pan_position).encode())
                arduino.write(b'Y')
                arduino.write(str(tilt_position).encode()) 
    return frame, boxes

modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

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
