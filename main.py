import cv2
from ultralytics import YOLO
import os

cap = cv2.VideoCapture("veh2.mp4")
import time

model = YOLO(r"yolov8s.pt")
folder_path = "./high_speed/"

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cy1 = 500
cy2 = 560

cx1 = 264
cx2 = 1077

cx3 = 140
cx4 = 1205

offset = 6

inside = []
outside = []

vh_down = {}

vh_up = {}
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.resize(frame, (1280, 720))
    img = frame.copy()
    img = cv2.resize(img,(1280,720))
    results = model.track(img.copy(), verbose=False,persist=True)
    # x = results[0].plot()

    if len(results[0].boxes) != 0:
        for box in results[0].boxes:
            # print(box)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            try:
                id = box.id.int()[0].item()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                # cv2.imshow("car",img[y1:y2,x1:x2])
                center_x = (x1+x2)//2
                center_y = (y1+y2)//2
                car = img.copy()
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.circle(img,(center_x,center_y),3,(255,0,255),-1)
                cv2.putText(img, f"{results[0].names[int(cls)]}, {id}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if cy1 < (center_y+offset) and cy1 > (center_y-offset):
                    cv2.circle(img,(center_x,center_y),3,(255,255,255),-1)
                    vh_down[id] = time.time()
                    
                if cy2 < (center_y+offset) and cy2 > (center_y-offset):
                    cv2.circle(img,(center_x,center_y),3,(255,255,255),-1)
                    vh_up[id] = time.time()
                    
                if id in vh_down:
                    if cy2 < (center_y+offset) and cy2 > (center_y-offset):
                        time_el = time.time()-vh_down[id]
                        speed = (10/time_el)*3.6

                        if results[0].names[int(cls)] == "car":
                            
                            cv2.circle(img,(center_x,center_y),3,(0,255,255),-1)
                            cv2.putText(img, f"{results[0].names[int(cls)]}, {id}, {speed:.0f} Km/h",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            if inside.count(id) == 0:
                                inside.append(id)
                                if int(speed) > 30:

                                    if not os.path.exists(folder_path):
                                        os.makedirs(folder_path)
                                    cv2.imwrite(f"./high_speed/car_incoming{id}_{speed:.0f}kmh.jpg",car[y1:y2,x1:x2])
                                
                if id in vh_up:
                    if cy1 < (center_y+offset) and cy1 > (center_y-offset):
                        time_el = time.time()-vh_up[id]
                        speed = (10/time_el)*3.6

                        if results[0].names[int(cls)] == "car":
                            
                            cv2.circle(img,(center_x,center_y),3,(0,255,255),-1)
                            cv2.putText(img, f"{results[0].names[int(cls)]}, {id}, {speed:.0f} Km/h",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            if outside.count(id) == 0:
                                outside.append(id)
                                if int(speed) > 30:

                                    if not os.path.exists(folder_path):
                                        os.makedirs(folder_path)
                                    cv2.imwrite(f"./high_speed/car_outgoing{id}_{speed:.0f}kmh.jpg",car[y1:y2,x1:x2])
            except:
                pass
    
    cv2.line(img,(cx1,cy1),(cx2,cy1),(0,0,255),2)
    cv2.line(img,(cx3,cy2),(cx4,cy2),(0,0,255),2)
    cv2.putText(img,f"In : {(len(inside))}",(300,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    cv2.putText(img,f"Out : {(len(outside))}",(300,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    cv2.imshow("RGB", img)
    # print(vh_down)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects
cap.release()
cv2.destroyAllWindows()
