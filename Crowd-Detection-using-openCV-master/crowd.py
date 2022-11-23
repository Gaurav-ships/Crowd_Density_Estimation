import numpy as np
import cv2
import os
import time
from scipy.spatial import distance as distance
import cmath
import imutils
import base64
import requests
import numpy as np
import matplotlib.pyplot as plt

# plt.axis([0, 10, 0, 1])


labelpath=r'C:\Users\Dell\Downloads\Crowd-Detection-using-openCV-master\Crowd-Detection-using-openCV-master\darknet\coco.names'
file=open(labelpath)
label=file.read().strip().split("\n")
label[0]

weightspath=r'C:\Users\Dell\Downloads\Crowd-Detection-using-openCV-master\Crowd-Detection-using-openCV-master\darknet\yolov3.weights'
configpath=r'C:\Users\Dell\Downloads\Crowd-Detection-using-openCV-master\Crowd-Detection-using-openCV-master\darknet\yolov3.cfg'

net=cv2.dnn.readNetFromDarknet(configpath,weightspath)

videopath=r'C:\Users\Dell\Downloads\Crowd-Detection-using-openCV-master\Crowd-Detection-using-openCV-master\video\run.mp4'

layer_names=net.getLayerNames()
ln = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

URL_EDUCATIONAL = "http://things.ubidots.com"
URL_INDUSTRIAL = "http://industrial.api.ubidots.com"
INDUSTRIAL_USER = True  # Set this to False if you are an educational user
TOKEN = ""  # Put here your Ubidots TOKEN
DEVICE = "detector"  # Device where will be stored the result
VARIABLE = "people"  # Variable where will be stored the result

"""def buildPayload(variable, value):
    return {variable: {"value": value}}"""
    
def buildPayload(variable, value, context):
    return {variable: {"value": value, "context": context}}

def convert_to_base64(image):
    image = imutils.resize(image, width=300)
    img_str = cv2.imencode('.png', image)[1].tostring()
    b64 = base64.b64encode(img_str)

    return b64.decode('utf-8')

def sendToUbidots(token, device, variable, value, context={}, industrial=True):
    # Builds the endpoint
    url = URL_INDUSTRIAL if industrial else URL_EDUCATIONAL
    url = "{}/api/v1.6/devices/{}".format(url, device)

    payload = buildPayload(variable, value,context)
    headers = {"X-Auth-Token": token, "Content-Type": "application/json"}

    attempts = 0
    status = 400
   
    while status >= 400 and attempts <= 5:
        req = requests.post(url=url, headers=headers, json=payload)
        status = req.status_code
        attempts += 1
        time.sleep(1)
        print(req)

    return req
def animate(i, xs, ys):

    # Read temperature (Celsius) from TMP102
    temp_c = round(tmp102.read_temp(), 2)

    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(temp_c)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('TMP102 Temperature over Time')
    plt.ylabel('Temperature (deg C)')

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
plt.show()


video=cv2.VideoCapture(videopath)
ret = video
init=time.time()
sample_time=5
if sample_time < 1:
        sample_time = 1
while(True):   
    ret,frame=video.read()
    if ret==False:
        print('Error running the file :(')
    frame=cv2.resize(frame, (640,440), interpolation =cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]
    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()

    boxes = []
    confidences = []
    classIDs = []
    center=[]
    output=[]
    count=0
    results=[]
    breach=set()
    
    h, w = frame.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
           
            confidence = scores[classID]
           
            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                center.append((centerX,centerY))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #color = [int(c) for c in colors[classIDs[i]]]
            if(label[classIDs[i]]=='person'):
                #people()
                cX=(int)(x+(y/2))
                cY=(int)(w+(h/2))
                center.append((cX,cY))
                res=((x,y,x+w,y+h),center[i])
                results.append(res) 
                dist=cmath.sqrt(((center[i][0]-center[i+1][0])**2)+((center[i][1]-center[i+1][1])**2))
                if(dist.real <100):
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255), 2)
                    cv2.circle(frame,center[i],4,(0,0,255),-1)
                    #cv2.line(frame, (center[i][0], center[i][1]), (center[i+1][0], center[i+1][1]), (0,0, 255), thickness=3, lineType=8)
                    count=count+1
        
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)
                    cv2.circle(frame,center[i],4,(0,255,0),-1)
                    count=count+1
        # plt.plot(count)
        # plt.pause(0.05)        
        
        #cv2.rectangle(frame,(startX, startY), (endX, endY),color, 2)
        #cv2.circle(frame,(cX,cY),4,color,-1)
        cv2.putText(frame,"Count: {}".format(count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2) 
        if time.time() - init >=sample_time:
            print("[INFO] Sending actual frame results")
            # Converts the image to base 64 and adds it to the context
            b64 = convert_to_base64(frame)
            context = {"image": b64}
            sendToUbidots(TOKEN, DEVICE,VARIABLE,count,context=context)
            init = time.time()
       

        
                    
    
    #cv2.putText(frame,"Violation: {}".format(count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (23,255,255), 1)      

    #frame_blob=makeblob(frame)
    # if frame!=0:
    cv2.imshow ('Frame',frame)
    # plt.show()
    # cv2.putText(frame,str(count), (200, frame/.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4) 
    
    # print(count)
    #cv2.VideoWriter(r'C:\Users\Lenovo\Desktop\Crowd Detection\video\walk2.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, (640,440)).write(frame) 
    
    if(cv2.waitKey(1)==ord('q')):
        break
video.release()
cv2.destroyAllWindows()