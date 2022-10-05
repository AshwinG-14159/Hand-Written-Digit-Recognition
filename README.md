# Hand-Written-Digit-Recognition
Created a system to detect handwritten digits from the webcam. This was my first attempt at implementing a neural network and was very successful from a 12th standard student's perspective. The code was crude and I did not know how to use a lot of libraries-hence much of the image pre processing is crude and done directly using a 2D numpy array form of the image.



To run the code:
Download the files, install the required libraries. Run the code.
First, it will create a neural network and train it. To do this it will download the MNIST dataset from an online source.
After training, the model will be saved on the device and it will ask for the number of images you want to take. 

It will open a gui for each image where you can chose the resolution(high needed if multiple digits expected in webcam image). Then click the 'click image' button and webcam will start up. Click space to take an Image and then escape to close the window. Then also close the tkinter window by clicking on the cross at the corner. The program may create more image popups(keep closing them as they may cause the program to pause) which will show the user their image as it proceeds throught the program. It shows how the image changes as it keeps getting processed.

The code will start analysing the image, it will take some time to run because the logic of the image processing is extremely crude and almost entirely done directly using numpy without any inbuilt image-processing libraries. 

Once the code has run, the final popup will show a processed image of the number(s) and the predictions.

In case of errors, watch out for version changes in libraries.



