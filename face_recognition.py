import face_recognition
import time
import cv2

yüz_algılama = cv2.CascadeClassifier('Data/haarcascades/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_DUPLEX
isimler = []
tanımlıyüzler = []
video = cv2.VideoCapture(0)
artan = 0
while True:
    _, kare = video.read()
    grikare = cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    yüz1 = yüz_algılama.detectMultiScale(grikare, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    artan += 1
    if type(yüz1) != tuple and artan > 10:
        xa = True
        while True:
            for (x,y,w,h) in yüz1:
                cv2.rectangle(kare,(x,y),(x+w,y+h),(255,255,0),2)
            if xa == True:
                giriş = input("Adını girin: ")
                xa = False
            try:
                ayüz = face_recognition.face_encodings(kare)[0]
                tanımlıyüzler.append(ayüz)
                isimler.append(giriş)   
            except:
                print("Olmadı...")
                print(kare.shape)
                print(type(kare))
                time.sleep(5)
                break
            cv2.putText(kare,"Cikmak icin 'ESC' tusuna basin...",(0,20),font,0.4,(255,255,0),1)
            cv2.imshow("Algilama Sistemi",kare)
            if cv2.waitKey(0) & 0xFF == 27:
                    break
    if cv2.waitKey(0) & 0xFF == 27:
            break
video.release()
cv2.destroyAllWindows()