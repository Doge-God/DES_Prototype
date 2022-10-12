import cv2
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    
    foundBarcodes = decode(img)

    if not foundBarcodes:
        pass
    else:
        for barcode in foundBarcodes:
            #barcode rectangle loc.
            (x,y,w,h) = barcode.rect
            #cv2.rectangle(img, (x-10, y-10),(x + w+10, y + h+10),
            #                (255, 0, 0), 2)

            print(barcode.polygon[0])
    
    cv2.imshow("test window", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()