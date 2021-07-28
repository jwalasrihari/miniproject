import cv2
import numpy as np

import matplotlib.pyplot as plt
from math import sqrt,exp
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)



import tkinter as tk

from tkinter import filedialog as fd 
from tkinter import ttk

address=""
gui = tk.Tk(className='Filters')
gui.geometry("380x550")
gui.configure(bg='#000000')


              
              
def acknowledge():
    
    global address
    address = fd.askopenfilename() 
    print(address)
    tk.messagebox.showinfo('Info','Image is uploaded')
    
errmsg = 'Error!'
tk.Button(text='Upload your image',font="Times 11 bold",command=acknowledge, width=30,height=1).pack(pady = 10, padx = 100)




def display(title,*images):
    
    length=len(images)
    
    if(length>=3):
        fig = plt.figure(figsize=(20,20), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(20,20), constrained_layout=False)
    # plt.xticks([]),plt.yticks([]) can be used to hide axis
    
    
    len_str=str(length)
    for i in range(1,length+1):
        temp=int('1'+len_str+str(i))
        plt.subplot(temp),plt.imshow(images[i-1], "gray"), plt.title(title[i-1]),plt.xticks([]),plt.yticks([])
    
    
    # Toplevel object which will 
    # be treated as a new window
    newWindow = tk.Toplevel(gui)
  
    # sets the title of the
    newWindow.title("output")
  
    # sets the geometry of toplevel
    if(length>3):
         # sets the title of the
        newWindow.title("Frequency Domain Filters")
        newWindow.geometry("800x200")
    else:
         # sets the title of the
        newWindow.title("Spatial Domain Filters")
        newWindow.geometry("600x300")
    
    newWindow.configure(bg='#ffffff')
                        
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,master = newWindow)  
    canvas.draw()
  
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,newWindow)
    toolbar.update()
  
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()
    
    
#========================spatial domain========================================== 
    
def convolve_linear(X, F):
    # height and width of the imagezz
    X_height = X.shape[0]
    X_width = X.shape[1]
    
    # height and width of the filter
    F_height = F.shape[0]
    F_width = F.shape[1]
    H = (F_height - 1) // 2
    W = (F_width - 1) // 2

    #output numpy matrix with height and width
    out = np.zeros((X_height-(2*H), X_width-(2*W)))
    #iterate over all the pixel of image X
    for i in np.arange(H, X_height-H):
        for j in np.arange(W, X_width-W):
            sum = 0
            #iterate over the filter
            for k in np.arange(-H, H+1):
                for l in np.arange(-W, W+1):
                    #get the corresponding value from image and filter
                    a = X[i+k, j+l]
                    w = F[H+k, W+l]
                    sum += (w * a)
            out[i-1,j-1] = sum
    #return convolution  
    return out
#-----------------------linear filter-------------------------------------------
def mean():
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    image=cv2.imread(temp,0)
    #0 incidcates read in gray scale by default
    k= np.ones((3,3),np.float32)/9
    processed_image = convolve_linear(image,k)
    title=["input image","mean filter output"]
    display(title,image,processed_image)

def gaussian():
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    image=cv2.imread(temp,0)
    #0 incidcates read in gray scale by default
    k= np.array([[1,2,1],[2,4,2],[1,2,1]],np.float32)/16
    processed_image = convolve_linear(image,k)
    title=["input image","gaussian filter output"]
    display(title,image,processed_image)
    
def laplacian(): 
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    image=cv2.imread(temp,0)
    
    k= np.array([[0,1,0],[1,-4,1],[0,1,0]]) 
    processed_image = convolve_linear(image,k)
    title=["input image","laplacian filter output"]
    display(title,image,processed_image)

def sobel():
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    image=cv2.imread(temp,0)
    sob_img_x=sobel_x(image)
    sob_img_y=sobel_y(image)
    
    sob_out = np.sqrt(np.power(sob_img_x, 2) + np.power(sob_img_y, 2))
    # mapping values from 0 to 255
    processed_image= (sob_out / np.max(sob_out)) * 255
    title=["input image","sobel filter output"]
    display(title,image,processed_image)

def sobel_x(img):
    k= np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) 
    sob_x = convolve_linear(img,k)
    return sob_x

def sobel_y(img):
    k= np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) 
    sob_y = convolve_linear(img,k)
    return sob_y

#----------------------Non-Linear filters---------------------------------------
def median_3(): 
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    image=cv2.imread(temp,0)
   
    sha = image.shape
    processed_image=image.copy()
   
    for i in range(1, sha[0] - 1):
        for j in range(1, sha[1] - 1):

            lst = []
            for x in range(i-1, i + 2):
                for y in range(j-1, j+2):
                    lst.append( image[x][y] )


            lst.sort()

            processed_image[i][j] = lst[4]
    title=["input image","median_3 filter output"]
    display(title,image,processed_image)


def median_5():

    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    image=cv2.imread(temp,0)
   
   
    sha = image.shape
    processed_image=image.copy()
   
    
    for i in range(1, sha[0] - 2):
        for j in range(1, sha[1] - 2):

            lst = []
            for x in range(i-2, i + 3):
                for y in range(j-2, j+3):
                    lst.append( image[x][y] )
            
            lst.sort()

            processed_image[i][j] = lst[12]
    title=["input image","median_5 filter output"]
    display(title,image,processed_image)
    
def min_filter():

    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    image=cv2.imread(temp,0)
   
    sha = image.shape
    processed_image=image.copy()
   
    for i in range(1, sha[0] - 1):
        for j in range(1, sha[1] - 1):

            lst = []
            for x in range(i-1, i + 2):
                for y in range(j-1, j+2):
                    lst.append( image[x][y] )
            
            lst.sort()

            processed_image[i][j] = lst[0]
    title=["input image","min filter output"]
    display(title,image,processed_image)
    
def max_filter():
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    image=cv2.imread(temp,0)
    
    sha = image.shape
    processed_image=image.copy()
    
    
    for i in range(1, sha[0] - 1):
        for j in range(1, sha[1] - 1):

            lst = []
            for x in range(i-1, i + 2):
                for y in range(j-1, j+2):
                    lst.append( image[x][y] )
            #sort the values

            lst.sort()

            processed_image[i][j] = lst[8]
    title=["input image","max filter output"]
    display(title,image,processed_image)
    


#==============================frequency domain================================
    
#-----------------------------frequency domain filters calculation-------------------------

def dist(p1,p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def idealLPFilter(D0,img_shape):
    filter = np.zeros(img_shape[:2])
    r, c = img_shape[:2]
    center = (r/2,c/2)
    for x in range(c):
        for y in range(r):
            if dist((y,x),center) < D0:
                filter[y,x] = 1
    return filter

def idealHPFilter(D0,img_shape):
    filter = np.ones(img_shape[:2])
    r, c = img_shape[:2]
    center = (r/2,c/2)
    for x in range(c):
        for y in range(r):
            if dist((y,x),center) < D0:
                filter[y,x] = 0
    return filter

def butterworthLPFilter(D0,img_shape,n):
    filter = np.zeros(img_shape[:2])
    r, c = img_shape[:2]
    center = (r/2,c/2)
    for x in range(c):
        for y in range(r):
            filter[y,x] = 1/(1+(dist((y,x),center)/D0)**(2*n))
    return filter

def butterworthHPFilter(D0,img_shape,n):
    filter = np.zeros(img_shape[:2])
    r, c = img_shape[:2]
    center = (r/2,c/2)
    for x in range(c):
        for y in range(r):
            filter[y,x] = 1-1/(1+(dist((y,x),center)/D0)**(2*n))
    return filter

def gaussianLPFilter(D0,img_shape):
    filter = np.zeros(img_shape[:2])
    r, c = img_shape[:2]
    center = (r/2,c/2)
    for x in range(c):
        for y in range(r):
            filter[y,x] = exp(((-dist((y,x),center)**2)/(2*(D0**2))))
    return filter

def gaussianHPFilter(D0,img_shape):
    filter = np.zeros(img_shape[:2])
    r, c = img_shape[:2]
    center = (r/2,c/2)
    for x in range(c):
        for y in range(r):
            filter[y,x] = 1 - exp(((-dist((y,x),center)**2)/(2*(D0**2))))
    return filter


#----------------------------------main----------------------------------------
    





def idealLowPass():
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    LowPassCenter = center * idealLPFilter(50,img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    
    title=["Original Image","Spectrum","Centered Spectrum","Centered * LP Filter","Decentralize","Ideal Low Pass"]
    display(title,img,np.log(1+np.abs(original)),np.log(1+np.abs(center)),np.log(1+np.abs(LowPassCenter)),np.log(1+np.abs(LowPass)),np.abs(inverse_LowPass))
        



def idealHighPass():

    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    HighPassCenter = center * idealHPFilter(50,img.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    
    title=["Original Image","Spectrum","Centered Spectrum","Centered * HP Filter","Decentralize","Ideal High Pass"]
    display(title,img,np.log(1+np.abs(original)),np.log(1+np.abs(center)),np.log(1+np.abs(HighPassCenter)),np.log(1+np.abs(HighPass)),np.abs(inverse_HighPass))
        

   
def butterworthLowPass():

    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    LowPassCenter = center * butterworthLPFilter(50,img.shape,10)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    
    title=["Original Image","Spectrum","Centered Spectrum","Centered * LP Filter","Decentralize","butterworth Low Pass"]
    display(title,img,np.log(1+np.abs(original)),np.log(1+np.abs(center)),np.log(1+np.abs(LowPassCenter)),np.log(1+np.abs(LowPass)),np.abs(inverse_LowPass))
        

    
def butterworthHighPass():

    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    HighPassCenter = center * butterworthHPFilter(50,img.shape,10)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    
    title=["Original Image","Spectrum","Centered Spectrum","Centered * HP Filter","Decentralize","butterworth High Pass"]
    display(title,img,np.log(1+np.abs(original)),np.log(1+np.abs(center)),np.log(1+np.abs(HighPassCenter)),np.log(1+np.abs(HighPass)),np.abs(inverse_HighPass))
        
#butterworthHighPass()
    
def gaussianLowPass():

    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    LowPassCenter = center * gaussianLPFilter(50,img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    
    title=["Original Image","Spectrum","Centered Spectrum","Centered * LP Filter","Decentralize","Gaussian Low Pass"]
    display(title,img,np.log(1+np.abs(original)),np.log(1+np.abs(center)),np.log(1+np.abs(LowPassCenter)),np.log(1+np.abs(LowPass)),np.abs(inverse_LowPass))
  
    
    
def gaussianHighPass():

    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    HighPassCenter = center * gaussianHPFilter(50,img.shape)
    HighPass = np.fft.ifftshift(HighPassCenter)
    inverse_HighPass = np.fft.ifft2(HighPass)
    
    title=["Original Image","Spectrum","Centered Spectrum","Centered * HP Filter","Decentralize","gaussian High Pass"]
    display(title,img,np.log(1+np.abs(original)),np.log(1+np.abs(center)),np.log(1+np.abs(HighPassCenter)),np.log(1+np.abs(HighPass)),np.abs(inverse_HighPass))
        

    
#------------------------all filter at one place-------------------------------

def fre_smooth_filters():
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    ideal_low=idealLPFilter(50,img.shape)
    butter_low = butterworthLPFilter(50,img.shape,10)
    gaussian_low= gaussianLPFilter(50,img.shape)
    title=["Ideal low pass","butterworth low pass","gaussian low pass"]
    display(title,ideal_low,butter_low,gaussian_low)
    
    
    
def fre_sharpen_filters():
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    ideal_high=idealHPFilter(50,img.shape)
    butter_high = butterworthHPFilter(50,img.shape,10)
    gaussian_high= gaussianHPFilter(50,img.shape)
    title=["Ideal high pass","butterworth high pass","gaussian high pass"]
    display(title,ideal_high,butter_high,gaussian_high)



def applied_smoothing():
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    
    ideal_center=center*idealLPFilter(50,img.shape)
    butter_center = center*butterworthLPFilter(50,img.shape,10)
    gaussian_center= center*gaussianLPFilter(50,img.shape)
    
    ideal_lowpass=np.fft.ifftshift(ideal_center)
    butter_lowpass=np.fft.ifftshift(butter_center)
    gaussian_lowpass=np.fft.ifftshift(gaussian_center)
    
    inver_ideal_lowpass=np.fft.ifft2(ideal_lowpass)
    inver_butter_lowpass=np.fft.ifft2(butter_lowpass)
    inver_gaussian_lowpass=np.fft.ifft2(gaussian_lowpass)
    
    title=["Ideal low pass","butterworth low pass","gaussian low pass"]
    display(title,np.abs(inver_ideal_lowpass),np.abs(inver_butter_lowpass),np.abs(inver_gaussian_lowpass))


def applied_sharpening():
    if(len(address)!=0):
        temp=address
    else:
        tk.messagebox.showinfo('Info','Please select Image')
        return;
    img = cv2.imread(temp, 0)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    
    ideal_center=center*idealHPFilter(50,img.shape)
    butter_center = center*butterworthHPFilter(50,img.shape,10)
    gaussian_center= center*gaussianHPFilter(50,img.shape)
    
    ideal_highpass=np.fft.ifftshift(ideal_center)
    butter_highpass=np.fft.ifftshift(butter_center)
    gaussian_highpass=np.fft.ifftshift(gaussian_center)
    
    inver_ideal_highpass=np.fft.ifft2(ideal_highpass)
    inver_butter_highpass=np.fft.ifft2(butter_highpass)
    inver_gaussian_highpass=np.fft.ifft2(gaussian_highpass)
    
    title=["Ideal low pass","butterworth low pass","gaussian low pass"]
    display(title,np.abs(inver_ideal_highpass),np.abs(inver_butter_highpass),np.abs(inver_gaussian_highpass))


#================================ktinter======================================
    
    

ttk.Separator(gui, orient='horizontal' ).pack(fill='x')
tk.Label(gui,text='Spatial domain filters',bg='black',fg='white',font = 'Times 12 italic').place(x=120,y=55)
tk.Button(text='Mean', command=mean,width=20).place(x = 25,y=90)
tk.Button(text='Gaussian', command=gaussian,width=20,height=1).place(x=200,y=90)
tk.Button(text='Laplacian', command=laplacian,width=20,height=1).place(x=25,y=135)
tk.Button(text='sobel', command=sobel,width=20,height=1).place(x=200,y=135)
tk.Button(text='median_3', command=median_3,width=20,height=1).place(x=25,y=180)
tk.Button(text='median_5', command=median_5,width=20,height=1).place(x=200,y=180)
tk.Button(text='Min_filter', command=min_filter,width=20,height=1).place(x=25,y=225)
tk.Button(text='max_filter', command=max_filter,width=20,height=1).place(x=200,y=225)



tk.Label(gui,text='Frequency domain filters',bg='black',fg='white',font = 'Times 12 italic').place(x=120,y=265)
tk.Button(text='Ideal Low Pass', command=idealLowPass,width=20,height=1).place(x = 25,y=305)
tk.Button(text='Ideal High Pass', command=idealHighPass,width=20,height=1).place(x= 200, y = 305)

tk.Button(text='Butterworth Low Pass', command=butterworthLowPass,width=20,height=1).place(x = 25,y = 350)
tk.Button(text='Butterworth High Pass', command=butterworthHighPass,width=20,height=1).place(x = 200,y = 350)


tk.Button(text='Gaussain low pass', command=gaussianLowPass,width=20,height=1).place(x = 25,y = 395)
tk.Button(text='Gaussain High pass', command=gaussianHighPass,width=20,height=1).place(x = 200,y = 395)

tk.Button(text='frequency low pass filters', command=fre_smooth_filters,width=20,height=1).place(x = 25,y = 440)
tk.Button(text='frequency high pass filters', command=fre_sharpen_filters,width=20,height=1).place(x= 200,y = 440)

tk.Button(text='All 3 smoothening', command=applied_smoothing,width=20,height=1).place(x = 25, y= 485)
tk.Button(text='All 3 sharenping', command=applied_sharpening,width=20,height=1).place(x=200, y = 485)
tk.mainloop()