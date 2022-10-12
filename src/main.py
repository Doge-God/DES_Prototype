from ctypes import alignment
from itertools import product
import cv2
import gooeypie as gp
import json
from multiprocessing.pool import ThreadPool
from pyzbar.pyzbar import decode
import numpy as np

def scanBarCode():
    cap = cv2.VideoCapture(0)

    bar_code_data = None

    while True:
        ret, img = cap.read()
        foundBarcodes = decode(img)

        if not foundBarcodes:
            pass
        else:
            for barcode in foundBarcodes:
                #barcode rectangle loc.
                (x,y,w,h) = barcode.rect
                cv2.rectangle(img, (x-10, y-10),(x + w+10, y + h+10),
                                (255, 0, 0), 2)
                
                if (barcode.type == 'EAN13'): 
                    bar_code_data = int(str(barcode.data, 'utf-8'))
                else:
                    print("Not EAN13")

        cv2.imshow("Barcode Scanner", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) and bar_code_data != None:
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    return bar_code_data

def scanImg():

    item_name = None

    cap = cv2.VideoCapture(0)
    thres = 0.5 # Threshold to detect object
    nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,280) #width 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,120) #height 
    cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #brightness 

    classNames = []
    with open('coco.names','r') as f:
        classNames = f.read().splitlines()

    font = cv2.FONT_HERSHEY_PLAIN
    #font = cv2.FONT_HERSHEY_COMPLEX
    Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

    weightsPath = "frozen_inference_graph.pb"
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=thres)

        if len(classIds) != 0:   #
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 200, 0), thickness=1)
                cv2.putText(img, classNames[classId-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,200,0), 2)

        cv2.imshow("AI Recognition",img)
        
        if classNames[classId-1] in recognisable_food:
            item_name = classNames[classId-1]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) and item_name != None:
            break
        
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    return item_name


#create thread for scanner
pool = ThreadPool(processes=1)

#create UI object
app = gp.GooeyPieApp('Fridge App')

#Load product data list
dataFile = open('product_info.json')
productData = json.load(dataFile)
recognisable_food = ['banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake']
dataFile.close()

#init list for stored product
###IMPORTANT: for now only storing in memory, NO persistent data storage set up###
storedProduct = []

#test function, ignore for production
def test(event):
    print('test')

def updateList(event):
    productList.clear()
    for prod in storedProduct:
        productList.add_row([prod[1], prod[2]])

def addProduct(event):
    async_data = pool.apply_async(scanBarCode,())
    return_data = async_data.get()
    for entry in productData:
        if (entry['code'] == return_data):
            if (([entry['code'], entry['name'], entry['expireInDays']]) in storedProduct):
                app.alert('Oops','This product is already in fridge.','info')
                return
            else:
                storedProduct.append([entry['code'], entry['name'], entry['expireInDays']])
                updateList(event)
                return
    #reach here if no matching data found in dataset
    app.alert('Oops','This product is not supported.','info')

def takeProduct(event):
    async_data = pool.apply_async(scanBarCode,())
    return_data = async_data.get()
    for entry in storedProduct:
        if (entry[0] == return_data):
            storedProduct.remove(entry)
            updateList(event)
            return
    #reach here if no matching data found in dataset
    app.alert('Oops','This product is not stored.','info')

def addProductImgRec(event):
    async_data = pool.apply_async(scanImg,())
    return_data = async_data.get()
    for entry in productData:
        if (entry['name'] == return_data):
            if (([entry['code'], entry['name'], entry['expireInDays']]) in storedProduct):
                app.alert('Oops','This product is already in fridge.','info')
                return
            else:
                storedProduct.append([entry['code'], entry['name'], entry['expireInDays']])
                updateList(event)
                return
    #reach here if no matching data found in dataset
    app.alert('Oops','This product is not supported.','info')

def takeProductImgRec(event):
    async_data = pool.apply_async(scanImg,())
    return_data = async_data.get()
    for entry in storedProduct:
        if (entry[1] == return_data):
            storedProduct.remove(entry)
            updateList(event)
            return
    #reach here if no matching data found in dataset
    app.alert('Oops','This product is not stored.','info')

productList = gp.Table(app,['NAME', 'EXPIRE IN (DAYS)'])
productList.set_column_widths(400,200)
productList.height = 10
productList.set_column_alignments('center','center')

bar_lbl = gp.Label(app, 'Barcode:')
add_btn = gp.Button(app, 'Put Item', addProduct)
take_btn = gp.Button(app, 'Take Item', takeProduct)

imgRec_lbl = gp.Label(app, 'AI Recognition:')
add_imgRec_btn = gp.Button(app, 'Put Item', addProductImgRec)
take_imgRec_btn = gp.Button(app, 'Take Item', takeProductImgRec)

app.width = 600
app.set_grid(7, 3)

app.add(bar_lbl, 1, 1, align = 'center')
app.add(add_btn, 1, 2, align = 'left', fill = True)
app.add(take_btn, 1, 3, alight = 'right', fill = True)

app.add(imgRec_lbl, 2, 1, align = 'center')
app.add(add_imgRec_btn, 2, 2, align = 'left', fill = True)
app.add(take_imgRec_btn, 2, 3, alight = 'right', fill = True)

app.add(productList, 3, 1, column_span = 3, fill = True)
app.run()

