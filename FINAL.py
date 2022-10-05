#%%
#importing required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from resizeimage import resizeimage
import cv2
from PIL import Image
from tkinter import *
import tkinter.font as font

#colouring an image
def colourit(arr):
    shape = np.shape(arr)
    i2 = np.zeros(tuple(shape + (3,)))
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            i2[i][j] = arr[i][j]
    
    return i2

#colouring a set of images
def compileit(imgs):
    arr = np.zeros(tuple(np.shape(imgs)+(3,)))
    for i in range(len(imgs)):
        arr[i] = colourit(imgs[i])
    return arr

#Training a machine learning model

#retrieving mnist data
mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0 #normalising data

x_train = compileit(x_train)
x_test = compileit(x_test)

#forming the model
model = keras.Sequential()


model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(56, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Training the model
history = model.fit(x_train, y_train, epochs=4, 
                    validation_data=(x_test, y_test))

#testing and displaying model accuracy
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(test_acc)
print('Model Trained.')

#saving the trained model
model.save('cnnmodel.h5')
model_json = model.to_json()
with open("cnnmodel.txt", "w") as json_file:
         json_file.write(model_json)
print('Model saved')        
         

from tensorflow.keras.models import model_from_json

#loading tensorflow module
def load_model():
    json_file = open('cnnmodel.txt', 'r')
    model_json = json_file.read()
    model1 = model_from_json(model_json)
    model1.load_weights("cnnmodel.h5")
    return model1

model1 = load_model()
print('Model loaded')

path=''

#changing image dimensions i.e blurring the image by averaging out pixels
def blur2(arr,req_v,req_h):
    ratio_v = len(arr)//req_v
    ratio_h = len(arr[0])//req_h
    l = []
    n = 0
    for i in range(req_v):
        l2 = []
        for j in range(len(arr[0])):
            t = n
            s = 0
            for k in range(ratio_v):
                s += arr[t][j]
                t += 1
            l2 += [s/ratio_v]
        l.append(l2)
        n += ratio_v
    lf = []
    for i in l:
        t = 0
        l2 = []
        for j in range(req_h):
            s = 0
            for k in range(ratio_h):
                s += i[t]
                t += 1
            l2 += [s/ratio_h]
        lf.append(l2)
    return np.array(lf)

#cleaning the image by removing pixels with a very light shade of gray
def sharpen(arr,extent = 0.46):
    l = []
    cor = np.zeros(np.shape(arr))
    cor = list(cor)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if float(arr[i][j]) < extent:
                pass
            else:
                cor[i][j] = arr[i][j]
    return np.array(cor)

#cropping image to certain size, coordinates a,b,c,d
def crop(arr,a,b,c,d):
    arr2 = np.zeros((b-a+1,d-c+1))
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if i<= b and i >= a:
                if j <= d and j>= c:
                    arr2[i-a][j-c] = arr[i][j]
    return arr2

#looking for a point at the boundary of a written digit
def findnewpoint(arr,n):
    l = []
    for i in range(0,len(arr),n):
        for j in range(1,len(arr[i])-1):
            if (arr[i][j] == 0 and arr[i][j+1] != 0) or (arr[i][j] == 0 and arr[i][j-1] != 0):
                l.append([i,j])
                break
    for i in range(0,len(arr[0]),n):
        for j in range(1,len(arr)-1):
            if (arr[j][i] == 0 and arr[j+1][i] != 0) or (arr[j][i] == 0 and arr[j-1][i] != 0):
                l.append([j,i])
                break
    return l

#finding a dark point
def finddark(arr):
    l = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] != 0:
                l.append([i,j])
    return l

#forming a list of all points
def findpoints(arr):
    l = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            l.append([i,j])
    return l

#finding a list of points on the boundary of a given point 
def boundarypoints(arr,x,y,center = False):
    l = []
    for i in range(-1,2):
        for j in range(-1,2):
            l.append([x+i,y+j])
    if center == False:
        l.pop(4)
    return l

#checking if points are neighbouring
def neighbouring(x1,y1,x2,y2):
    if x2 == x1 and (y1-y2 == 1 or y2 - y1 == 1):
        return True
    elif y2 == y1 and (x1-x2 == 1 or x2 - x1 == 1):
        return True
    else:
        return False

#getting a point to move to from a given point, along the written digit
def getmovablepoint(arr,l):
    global restricted
    for i in l:
        for j in l:
            if arr[i[0]][i[1]] == 0 and arr[j[0]][j[1]] !=0 :
                if neighbouring(i[0],i[1], j[0],j[1]):
                    if i not in restricted:
                        restricted.append(i)
                        return i
    return None

#creating a comlete boundary around a digit
def createboundary(arr,pt):
    bdpts = boundarypoints(arr, pt[0], pt[1])
    movable = getmovablepoint(arr, bdpts)
    return movable

#finding a box around the boundary of a digit so as to separate it
def findno(arr,n):
    global points
    pt = points[n]
    pt = createboundary(arr,pt)
    pthmin, pthmax = len(arr), 0
    ptvmin, ptvmax = len(arr[0]), 0
    if pt != None:
        while pt != None:

            if pt[0] > pthmax:
                pthmax = pt[0]
            if pt[0] < pthmin:
                pthmin = pt[0]
            if pt[1] > ptvmax:
                ptvmax = pt[1]
            if pt[1] < ptvmin:
                ptvmin = pt[1]
            pt = createboundary(arr,pt)
            
        return(pthmin,pthmax,ptvmin,ptvmax)
    else:
        return(0,len(arr),0,len(arr[0]))   

#drawing the box on the image
def drawbox(arr,n):
    a,b,c,d = findno(arr,n)
    t = crop(arr,a,b,c,d)

    return t,c,d

#drawing boxes around all digits by finding all objects in an image
def findobjects(arr,n=5):
    l = []
    global points
    points = findnewpoint(arr,n)
    t = len(points)
    for i in range(t):
        print('Continuing....')
        if points[i] not in restricted:
            t,a,b = drawbox(arr,i)
            l.append([t,a,b])

    return l

#darkening points around a given point to emphasize a thin region in a digit
def darken(arr,x,y):
    bdpts = boundarypoints(arr, x, y, center = True)
    for i in bdpts:
        arr[i[0]][i[1]] = (arr[i[0]][i[1]] + 0.1)/2
    
    return arr

#duplicating an image(numpy array format)
def duplicate(arr):
    arr2 = np.zeros(np.shape(arr))
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j] != 0:
                arr2[i][j] = arr[i][j]
    return arr2

#darkening the regions around all written objects
def bold(arr,a):
    global points
    arr2 = duplicate(arr)
    points = findpoints(arr)
    print('Step 2 in progress')
    for i in points:
        for j in points:
            if arr[i[0]][i[1]] == 0 and arr[j[0]][j[1]] !=0:
                if neighbouring(i[0], i[1], j[0], j[1]):
                    arr2[i[0]][i[1]] = a
    print('Step 3')
    return arr2

#second layer of cleaning to remove unnecessary grey pixels
def clean(arr,n):
    cor = np.zeros((28,28))
    cor = list(cor)
    for i in range(28):
        for j in range(28):
            if float(arr[i][j]) < n:
                pass
            else:
                cor[i][j] = arr[i][j]
    return np.array(cor)

#exaggerating the difference between grey and black pixels
def exagerate(arr):
    arr = arr*arr*arr*arr*2
    return arr

#shrinking a numpy array
def shrink(arr):
    s = np.shape(arr)[0:2]
    arr2 = np.zeros((s))
    for i in range(len(arr2)):
        for j in range(len(arr2[0])):
            arr2[j][i] = (arr[j][i][0] + arr[j][i][1] + arr[j][i][2])/3
    return arr2

#preparing an image for recognition
def prepare(img,x,y):
    img = Image.fromarray(np.uint8((1-img) * 255) , 'L')
    img = resizeimage.resize_contain(img, [x, y])
    x = np.array(img)
    x = x/255
    x = shrink(x)
    x = 1-x
    return x

#forming a white padding around an image
def pad(arr,x,y):
    arr2 = np.zeros((len(arr)+x, len(arr[0])+y ) )
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            arr2[i+x][j+y] = arr[i][j]
    return arr2

#capturing an image from the camera
def capture():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Camera")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.rectangle(frame,(220,220),(340,340),(255,0,0),2)
        cv2.imshow("Camera", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(imageval)
            frame = frame[220:340,220:340]
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            global path
            path=img_name
    cam.release()
    cv2.destroyAllWindows()

print('Functions ready to use.')    


#MAIN CODE   
from PIL import Image 

#setting the complexity for an image given by user
def complexity(r):
    global complexval
    complexval = r

#creation and running of the window with buttons
def button():   
 tkWindow = Tk()  
 tkWindow.title('Image Recognition System')
 tkWindow.configure(background='Black')
 tkWindow.geometry('400x200') 
 myFont = font.Font(family="Times New Roman", size=9)
 button2 = Button(tkWindow, text = 'Low accuracy',padx='10',pady='10',command = complexity(30),bg='Yellow')
 button2.place(x='100',y='20')
 button2['font'] = myFont
 button3 = Button(tkWindow,text = 'Med accuracy',padx='10',pady='10',command = complexity(50),bg='green')  
 button3.place(x='100',y='70')
 button3['font'] = myFont
 button4 = Button(tkWindow,text = 'High accuracy',padx='10',pady='10',command = complexity(80),bg='Cyan') 
 button4.place(x='100',y='120')
 button4['font'] = myFont
 button = Button(tkWindow, text = 'Click Image',padx='10',pady='10', command = capture)
 button.place(x='250',y='70')
 button['font'] = myFont
 tkWindow.mainloop()


#printing the value that has been predicted in a window
def image(prediction):
 root = Tk()
 root.title('Prediction')
 canvas = Canvas(root, width = 200, height = 200)
 canvas.pack()
 img = PhotoImage(file=path)  
 canvas.create_image(100,100,image=img)
 w=Text(root,height=5,width=100)
 w.insert(INSERT,'The number in the image is {}'.format(prediction))
 w.pack()
 root.mainloop()



n = int(input('Enter number of images: '))
complexval = 50
for i in range(n):
    global imageval
    imageval = i
    restricted = []
    complexval = 40
    button()#taking image input
    img = cv2.imread('opencv_frame_{}.png'.format(i), -1)
    
    # image cleaning to remove shadows
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    cv2.imwrite('shadows_out{}.png'.format(i), result)
    cv2.imwrite('shadows_out_norm{}.png'.format(i), result_norm)
    
    im = Image.open('shadows_out_norm{}.png'.format(i)).convert('L')
    im = im.crop((3,3,117,117))
    im.save('shadows_out_norm{}.png'.format(i))
    
    #image editing to make it ready for recognition
    im = Image.open('shadows_out_norm{}.png'.format(i)).convert('L')
    im = im.resize((complexval,complexval))
    im = 255 - np.array(im)
    img = im/255
    plt.imshow(im,cmap = 'gray_r')
    plt.show()

    img = sharpen(img,0.2)
    plt.imshow(img,cmap = 'gray_r')
    plt.show()


    img = pad(img,5,5)
    print('Step 1 done')
    img = bold(img,.2)
    plt.imshow(img,cmap = 'gray_r')
    plt.show()
    
    #breaking image into smaller images of individual digits
    smallimages = list(findobjects(img,4))
    
    #removing duplicate images
    for i in range(1,len(smallimages)):
        t = smallimages[i]
        while n != 0 and t[2] < smallimages[n-1][1]:
            smallimages[n] = smallimages[n-1]
            smallimages[n-1] = t
            n -= 1
    
    
    #preparing a small image for recognition
    for i in smallimages:
        i[0] = pad(i[0],3,3)
        i[0] = prepare(i[0],28,28)
        i[0] = colourit(i[0])
    
    numberlist = []
    
    #recognising the digits
    for i in smallimages:
        prediction = model1.predict(np.array([i[0]]))
        predicti = np.argmax(prediction)
        print('prediction: ',predicti)
        numberlist += [predicti]
        plt.imshow(i[0],cmap = 'gray_r')
        plt.show()
    
    image(numberlist)


