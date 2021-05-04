import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
##########################################################################################
def ResizeToScreen(Im):

        h=Im.shape[0]
        w=Im.shape[1]
        r=np.min([600/h,1200/w])
        Im=cv2.resize(Im,(int(r*w),int(r*h)))
        return Im
########################################################################################
# def showFuckOPENCV(Im,txt=""):
#     cv2.destroyAllWindows()
#    # print("IM text")
#    # print(txt)
#     cv2.imshow(txt,ResizeToScreen(Im.astype(np.uint8)))
#   #  cv2.moveWindow(txt, 1, 1);
#     ch=cv2.waitKey()
#    # cv2.destroyAllWindows()
#     return ch
########################################################################################
def show(Im,txt=""):
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    if np.ndim(Im)==3:
        plt.imshow(Im[:, :, ::-1].astype(np.uint8))
    else:
        plt.imshow(Im.astype(np.uint8))
    plt.title(txt)
    plt.show()
########################################################################################
def trshow(Im,txt=""):
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.imshow((Im.data.cpu().numpy()).astype(np.uint8))
    plt.title(txt)
    plt.show()
#############################################################################3
def GreyScaleToRGB(Img):
    I=np.expand_dims(Img,2)
    rgb=np.concatenate([I,I,I],axis=2)
    return  rgb
################# #########################################################################
def ResizeToLimit(Im,MinSize,MaxSize,interpolation):
    #If images are to small or two big resize them
        h0,w0=Im.shape[:2]
        mxD=np.max([h0,w0])
        mnD=np.min([h0,w0])
        r=0
        if mxD>MaxSize:
            r=MaxSize/mxD
        if mnD<MinSize:
            r=MinSize/mnD
        if r>0:
            Im = cv2.resize(Im,(int(r*w0),int(r*h0)),interpolation=interpolation)
        return Im,  w0, h0
###########################################################################################


