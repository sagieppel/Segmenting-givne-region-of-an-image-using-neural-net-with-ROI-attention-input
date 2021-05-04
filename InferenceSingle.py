# Apply inference to single mask in single image.
#...............................Imports..................................................................
import os
import numpy as np
import Visuallization as vis
import cv2
import shutil
import json
import torch
##################################Input paramaters#########################################################################################
# ####################################################################################################
MinImSize=200 # min image height/width smaller images will be resize
MaxImSize=1000#  max image height/width smaller images will be resize

MinClassThresh=0.3 # Threshold for assigning class to image
MinInsPixels=250 # Min pixels in prediction smaller instances will not be registered
#********************************************************************************************************************************************


def InferSingleVessel(Net,ImagePath,VesMaskPath,OutDirInst,OutDirSemantic,OutDirInstDiplay="",OutDirSemanticDisplay="",ClassesToUse="",Display=False,UseGPU=False): # Run Predict on a single vessel in an image
    if not os.path.exists(OutDirInst): os.mkdir(OutDirInst)
    if not os.path.exists(OutDirSemantic): os.mkdir(OutDirSemantic)
    if not os.path.exists(OutDirInstDiplay) and OutDirInstDiplay!="": os.mkdir(OutDirInstDiplay)
    if not os.path.exists(OutDirSemanticDisplay) and OutDirSemanticDisplay!="": os.mkdir(OutDirSemanticDisplay)


    Image = cv2.imread(ImagePath)  # Load Image
    Img,  w0, h0=vis.ResizeToLimit(Image,MinImSize,MaxImSize,interpolation=cv2.INTER_NEAREST) # Resize if image to large or small
   # Img=Img[:,:,::-1]
    Img = np.expand_dims(Img, axis=0)
    VesMask = (cv2.imread(VesMaskPath, 0) > 0).astype(np.uint8) # Read vessel mask
    VesMask, w0, h0 = vis.ResizeToLimit(VesMask, MinImSize, MaxImSize, interpolation=cv2.INTER_NEAREST) # Resize if to big or small
    VsMsk=np.expand_dims(VesMask,axis=0)
    with torch.no_grad(): # Run Prediction
        ProbInst, LbInst, ProbSemantic, LbSemantic = Net.forward(Images=Img,ROI=VsMsk,TrainMode=False,PredictSemantic=True,PredictInstance = True,UseGPU=UseGPU) # Run net inference and get prediction
    Net.zero_grad()
    ################################Set Instance Class###########################################################################################################
    ################################Filter and Set Instance Class by comparing instances to semantic maps###########################################################################################################
    NumInst=0
    InstMasks={}
    InstClasses = {}
    InstProbClass = {}
    VesMask = cv2.resize(VesMask, (int(w0), int(h0)), interpolation=cv2.INTER_NEAREST)
    for InsMsk in LbInst:  # Find overlap between instance and semantic maps to deterine class
        InsMsk = InsMsk[0]#.data.cpu().numpy()
        SumAll=InsMsk.sum()
        if SumAll>MinInsPixels:
            NumInst+=1
            InstClasses[NumInst] = []
            InstProbClass[NumInst] = {}
            for nm in LbSemantic:
                if not nm in ClassesToUse: continue
                SemMask = LbSemantic[nm][0]#.data.cpu().numpy()
                Fract=float(((InsMsk*SemMask).sum()/SumAll).data.cpu().numpy()) # find overlap between instace and semantic maps
                if Fract>MinClassThresh: # If overlap exceed thresh assign class to instance
                    InstClasses[NumInst].append(nm)
                    InstProbClass[NumInst][nm]=Fract


            InstMasks[NumInst]=InsMsk.data.cpu().numpy().astype(np.uint8)
    ################################Write and Display###########################################################################################################
        # OutSemDir = OutDir+"/Semantic/"
        # OutInstanceDir = OutDir + "/Instance/"
        # OutSemDirOverlay = OutDir + "/SemanticOverlay/"
        # OutInstanceDirOverlay = OutDir + "/InstanceOverlay/"
        # if os.path.exists(OutDir): shutil.rmtree(OutDir)
        # os.mkdir(OutDir)
        # os.mkdir(OutSemDir)
        # os.mkdir(OutInstanceDir)
        # os.mkdir(OutSemDirOverlay)
        # os.mkdir(OutInstanceDirOverlay)
    #---------------Save instance categories-----------------------------------------------
        with open(OutDirInst+'/InstanceClassList.json', 'w') as fp: # List of classes for instance
            json.dump(InstClasses, fp)
        with open(OutDirInst+'/InstanceClassProbability.json', 'w') as fp: # Class probability  for instance
            json.dump(InstProbClass, fp)
        # cv2.imwrite(OutDir+"/Image.jpg",Image)
       # cv2.imwrite(OutDirInst + "/Mask.png",VesMask)
    # ------------------------DisplayInstances----------------------------------------

        I = Image.copy()
        I1 = I.copy()
        I1[:, :, 0][VesMask > 0] = 0
        I1[:, :, 1][VesMask > 0] = 0
        I1[:, :, 2][VesMask > 0] = 255
        if Display == True:
            vis.show(np.concatenate([I, I1], axis=1), "Vessel  ")

        for ins in InstMasks: # Save instance maps
            Mask=InstMasks[ins]
            Mask = cv2.resize(Mask, (int(w0), int(h0)),interpolation=cv2.INTER_NEAREST)
            if Mask.sum()==0: continue
            if nm == 'Vessel': continue
#-------------Save instance and display-------------------------------------------------------------
            I2 = I1.copy()
            I2[:, :, 0][Mask > 0] = 0
            I2[:, :, 1][Mask > 0] = 255

            cv2.imwrite(OutDirInst+"/"+str(ins)+".png",Mask.astype(np.uint8))
            Overlay=np.concatenate([I, I1, I2, vis.GreyScaleToRGB(Mask * 255)],axis=1).astype(np.uint8)
            if OutDirInstDiplay != "": # Instance maps overlay on image
                cv2.imwrite(OutDirInstDiplay + "/" + str(ins) + ".png", Overlay)
                cv2.imwrite(OutDirInstDiplay + "/" + str(ins) + "B.png", I2)
            if Display: vis.show(Overlay,"Instace:" + str(InstClasses[ins])+" "+str(InstProbClass[ins]))

    # ---------------Display  and save Semantic maps-----------------------------------------------------------------------------------------------
        I = Image.copy()
        for nm in LbSemantic:
            Mask = LbSemantic[nm][0].data.cpu().numpy()
            Mask = cv2.resize(Mask, (int(w0), int(h0)), interpolation=cv2.INTER_NEAREST)
            if Mask.sum() == 0: continue
            if nm == 'Vessel': continue
            I2 = I1.copy()
            I2[:, :, 0][Mask > 0] = 0
            I2[:, :, 1][Mask > 0] = 255

            cv2.imwrite(OutDirSemantic + "/" + nm + ".png", Mask.astype(np.uint8))
            Overlay = np.concatenate([I, I1, I2, vis.GreyScaleToRGB(Mask * 255)], axis=1)
            if OutDirInstDiplay!="": # Semantic Map overlay on image
                cv2.imwrite(OutDirSemanticDisplay + "/" + nm + ".png", Overlay)
                cv2.imwrite(OutDirSemanticDisplay + "/" + nm + "B.png", I2)
            if Display: vis.show(Overlay,"Semantic:" + nm)
