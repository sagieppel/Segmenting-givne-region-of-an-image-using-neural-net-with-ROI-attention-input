## Reader for labpics dataset



import numpy as np
import os
import cv2
import json
import threading
import ClassesGroups
import Visuallization as vis
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, MainDir=r"", MaxBatchSize=100,MinSize=250,MaxSize=1000,MaxPixels=800*800*5,MinMaskSize=1000, TrainingMode=True, IgnoreParts=True,IgnoreSurfacePhase=True, IgnoreScattered=True,ClassToUse=[]):
        self.ClassToUse=ClassToUse
        self.IgnoreParts = IgnoreParts
        self.IgnoreSurfacePhase = IgnoreSurfacePhase
        self.IgnoreScattered = IgnoreScattered
        self.MinMaskSize=MinMaskSize # mininimal vessel instance size in pixel (smaller will be ignored
        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
# ----------------------------------------Create list of annotations arranged by class--------------------------------------------------------------------------------------------------------------
        self.AnnList = [] # Image/annotation list
        self.AnnByCat = {} # Image/annotation list by class

        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(MainDir):
            self.AnnList.append(MainDir+"/"+AnnDir)

#------------------------------------------------------------------------------------------------------------

        print("done making file list Total=" + str(len(self.AnnList)))
        if TrainingMode:
            self.StartLoadBatchInstance() # Start loading instance  mask batch (multi threaded)
            self.StartLoadBatchSemantic() # Start loading semantic maps batch (multi threaded)
        self.AnnData=False
#############################################################################################################################
#############################################################################################################################
# Crop and resize image and annotation mask and ROI to feet batch size
    def CropResize(self,Img, InsMasks ,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        bbox= cv2.boundingRect(InsMasks['Vessel'].astype(np.uint8))
        [h, w, d] = Img.shape
#=================================================================================================
        if np.random.rand() < 0.3:
            Img = cv2.resize(Img, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)
            for nm in InsMasks:
                InsMasks[nm] = cv2.resize(InsMasks[nm], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)
            return Img, InsMasks
#====================================================================================================

        Wbox = int(np.floor(bbox[2]))  # Segment Bounding box width
        Hbox = int(np.floor(bbox[3]))  # Segment Bounding box height
        if Hbox == 0 or Wbox == 0:
            print("empty box")
            exit(0)
        if Wbox==0:  Wbox+=1
        if Hbox == 0: Hbox += 1

        Rs = np.max((Hb / h, Wb / w)) # Check if target size larger then corrent image size
        Bs = np.min((Hb / Hbox, Wb / Wbox)) # Check if target size smaller then bounding box
        if Rs > 1 or Bs<1 or np.random.rand()<0.2:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(np.max((h * Rs, Hb)))
            w = int(np.max((w * Rs, Wb)))
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            for nm in InsMasks:
                InsMasks[nm] = cv2.resize(InsMasks[nm], dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            bbox = (np.float32(bbox) * Rs.astype(np.float)).astype(np.int64)

 # =======================Crop image to fit batch size===================================================================================
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height

        if Wb > Wbox:
            Xmax = np.min((w - Wb, x1))
            Xmin = np.max((0, x1 - (Wb - Wbox)-1))
        else:
            Xmin = x1
            Xmax = np.min((w - Wb, x1 + (Wbox - Wb)+1))

        if Hb > Hbox:
            Ymax = np.min((h - Hb, y1))
            Ymin = np.max((0, y1 - (Hb - Hbox)-1))
        else:
            Ymin = y1
            Ymax = np.min((h - Hb, y1 + (Hbox - Hb)+1))

        if Ymax<=Ymin: y0=Ymin
        else: y0 = np.random.randint(low=Ymin, high=Ymax + 1)

        if Xmax<=Xmin: x0=Xmin
        else: x0 = np.random.randint(low=Xmin, high=Xmax + 1)

        # Img[:,:,1]*=Mask
        # misc.imshow(Img)

        Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
        for nm in InsMasks:
            InsMasks[nm] = InsMasks[nm][y0:y0 + Hb, x0:x0 + Wb]
            if not (InsMasks[nm].shape[0] == Hb and InsMasks[nm].shape[1] == Wb): InsMasks = cv2.resize(InsMasks[nm].astype(float),dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)
#------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb): Img = cv2.resize(Img, dsize=(Wb, Hb),interpolation=cv2.INTER_LINEAR)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img,InsMasks

######################################################Augmented Image##################################################################################################################################
    def Augment(self,Img,InsMasks,prob):
        Img=Img.astype(np.float)
        if np.random.rand()<0.5: # flip left right
            Img=np.fliplr(Img)
            for nm in InsMasks:
                InsMasks[nm]= np.fliplr(InsMasks[nm])

        # if np.random.rand()<0.0: # flip left up down
        #     Img=np.flipud(Img)
        #     AnnMap = np.flipud(AnnMap)
        #
        # if np.random.rand()<0.0: # Change from rgb to bgr
        #     Img = Img[..., :: -1]




        if np.random.rand() < prob: # resize
            r=r2=(0.4 + np.random.rand() * 1.6)
            if np.random.rand() < prob:
                r2=(0.5 + np.random.rand())
            h = int(Img.shape[0] * r)
            w = int(Img.shape[1] * r2)
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            for nm in InsMasks:
                  InsMasks[nm] = cv2.resize(InsMasks[nm], dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        # if np.random.rand() < prob/3: # Add noise
        #     noise = np.random.rand(Img.shape[0],Img.shape[1],Img.shape[2])*0.2+np.ones(Img.shape)*0.9
        #     Img *=noise
        #     Img[Img>255]=255
        #
        # if np.random.rand() < 0: # Gaussian blur
        #     Img = cv2.GaussianBlur(Img, (5, 5), 0)

        if np.random.rand() < prob:  # Dark light
            Img = Img * (0.5 + np.random.rand() * 0.65)
            Img[Img>255]=255

        if np.random.rand() < prob*0.2:  # GreyScale
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img,InsMasks
##################################################################################################################################################################
# ==========================Read image and instance mask and data for single image into the batch===============================================================================================
    def LoadNextInstance(self, pos, Hb, Wb):
# -----------------------------------select image-----------------------------------------------------------------------------------------------------

            AnnInd = np.random.randint(len(self.AnnList))
            InPath=self.AnnList[AnnInd]
            data = json.load(open(InPath+'/Data.json', 'r'))


            #print(Ann)
            Img = cv2.imread(InPath+"/Image.jpg")  # Load Image
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
            InsMasks={} # All instance masks
            InsData={} # Instance Properties
#------------------Select vessel and read relevant vessel mask-------------------------------------------------------------------------------------------------
            VesselsChecked=[] # List of vessels that was checked and found to small
            VesselList=list(data["Vessels"].keys())
            while(True): # Pick vessel
               if len(VesselsChecked)==len(list(VesselList)):
                        return self.LoadNextInstance(pos, Hb, Wb)
               ind=VesselList[np.random.randint(len(VesselList))]
               if ind in VesselsChecked: continue
               VesselsChecked.append(ind)
               Ves=data["Vessels"][ind]
               #if len(Ves['ContainVessels_Indx'])>0: continue
               if Ves['IsPart']: continue # Ignore vessels that are parts (Connectors and condensers)
               InsMasks["Vessel"] = (cv2.imread(InPath + Ves['MaskFilePath'], 0)>0).astype(np.uint8)  # Read vesse instance mask
               if InsMasks["Vessel"].sum()>self.MinMaskSize: break
#-----------------------------Read Vessel content ----------------------------------------------------------------------------------------
            for contInd in Ves['VesselContentAll_Indx']:  # Go over indexes of all materials and parts in vessel
                cont = data['MaterialsAndParts'][str(contInd)]  # read material instance data

                if (cont['IsPart'] and self.IgnoreParts) or (cont['IsOnSurface'] and self.IgnoreSurfacePhase) or (cont['IsScattered'] and self.IgnoreScattered) or (cont['IsScattered'] and cont['IsOnSurface']): continue

                if not  cont['ASegmentableInstance']: continue
                InsMasks[contInd] = (cv2.imread(InPath + cont['MaskFilePath'], 0)).astype(np.uint8)  # Read material mask
                InsData[contInd] = cont

#-----------------------------------Augment Crop and resize-----------------------------------------------------------------------------------------------------
            if np.random.rand()<0.5:   Img,InsMasks=self.Augment(Img,InsMasks,0.3)
            if not Hb==-1:
               Img, InsMasks = self.CropResize(Img, InsMasks, Hb, Wb)
#----------------------------------------Weight per pixel for loss less inportant classes or relations will get lower weight for loss--------------------------------------------------------------------------------------
            InsMasksWeighted={} # Weight per pixel for loss
            InsMasksNew={}

            for nm in InsMasks:
                if InsMasks[nm].sum()>0:
                    Fact = 1
                    if nm!="Vessel":
                        if InsData[nm]['IsPart']: Fact *= 0.5
                        if InsData[nm]['IsOnSurface']: Fact *= 0.5
                        if InsData[nm]['IsScattered']: Fact *= 0.5
                        if 'Solid' in InsData[nm]['All_ClassNames']: Fact *= 0.6 # solid instances are harder to define
                        InsMasksWeighted[nm]=np.zeros(InsMasks[nm].shape,dtype=np.float32)
                        InsMasksWeighted[nm][InsMasks[nm] == 1]=Fact*1  #not overlap
                        InsMasksWeighted[nm][InsMasks[nm] == 4] = Fact*1 # Inside other phase
                        InsMasksWeighted[nm][InsMasks[nm] == 5] = Fact*0.4 # Contain
                        InsMasksWeighted[nm][InsMasks[nm] == 6] = Fact*1 # Front
                        InsMasksWeighted[nm][InsMasks[nm] == 7] = Fact*0.3 # Beyond
                        InsMasksNew[nm]=((InsMasks[nm]*InsMasks["Vessel"])>0).astype(np.float32)
            # if AnnMask.sum()<900:
            #       return self.LoadNext(pos, Hb, Wb)
#----------------------Load predicion into batch-----------------------------------------------------------------------------------------------------------
            self.BROIInst[pos][InsMasks["Vessel"]>0]=1
            self.BMasksInst[pos] = InsMasksNew
            self.BMasksWeightedInst[pos] = InsMasksWeighted
            self.BImgInst[pos] = Img
            self.BInsDataInst[pos] =  InsData
##################################################################################################################################################################
# ==========================Read single semantic image into batch===============================================================================================
    def LoadNextSemantic(self, pos, Hb, Wb):
# -----------------------------------select image-----------------------------------------------------------------------------------------------------

            AnnInd = np.random.randint(len(self.AnnList))
            InPath=self.AnnList[AnnInd]
            data = json.load(open(InPath+'/Data.json', 'r'))
            #print(Ann)
            Img = cv2.imread(InPath+"/Image.jpg")  # Load Image
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
            Masks={} # All instance masks
#------------------Select vessel and read relevant vessel mask-------------------------------------------------------------------------------------------------
            VesselsChecked=[] # List of vessels that was checked and found to small
            VesselList=list(data["Vessels"].keys())
            ind=-1
            while(True): # Pick vessel
               if len(VesselsChecked)==len(list(VesselList)):
                        return self.LoadNextSemantic(pos, Hb, Wb)
               ind=VesselList[np.random.randint(len(VesselList))]
               if ind in VesselsChecked: continue
               VesselsChecked.append(ind)
               Ves=data["Vessels"][ind]
             #  if len(Ves['ContainVessels_Indx'])>0: continue
               if Ves['IsPart']: continue # Ignore vessels that are parts (Connectors and condensers)
               Masks["Vessel"] = (cv2.imread(InPath + Ves['MaskFilePath'], 0)>0).astype(np.uint8)  # Read vesse instance mask
               if Masks["Vessel"].sum()>self.MinMaskSize: break
#-----------------------------Read AllSemanticMaps ----------------------------------------------------------------------------------------
            SemanticDir=InPath+ r"/SemanticMaps/PerVessel/"+str(ind)+"/"
            for FileName in os.listdir(SemanticDir):
  # Go over indexes of all materials and parts in vessel

                CatName=FileName.replace(".png","")
                if (CatName not in self.ClassToUse): continue
                Masks[CatName] = (cv2.imread(SemanticDir+"/"+FileName)).astype(np.uint8)  # Read material mask

#-----------------------------------Augment Crop and resize-----------------------------------------------------------------------------------------------------

            if np.random.rand()<0.5:   Img,Masks=self.Augment(Img,Masks,0.3)
            if not Hb==-1:
               Img, Masks = self.CropResize(Img, Masks, Hb, Wb)
#----------------------------------------Weight per pixel for loss--------------------------------------------------------------------------------------
            MasksWeighted={} # Weight per pixel for loss
            MasksNew={}

            for nm in Masks:
                if Masks[nm].sum()>0:
                    Fact = 1
                    if  nm in ClassesGroups.PartClass:
                        Fact*=0.4
                    #if nm
                    if nm!="Vessel":
                        MasksWeighted[nm]=Masks[nm][:,:,0].astype(np.float32)*Fact
                      #  MasksWeighted[nm][Masks[nm] == 1]=Fact*1  #not overlap
                        MasksWeighted[nm][Masks[nm][:,:,2] == 20] = 0.4*Fact # On surface
                        MasksWeighted[nm][Masks[nm][:,:,2] == 30] = 0.4*Fact # Scattered
                        MasksWeighted[nm][Masks[nm][:,:,2] == 40] = 0.3*Fact #Scattered and on surface
                        if 40 in Masks[nm]:
                            rr=3
                       # MasksNew[nm]=((Masks[nm][:,:,0]*Masks["Vessel"])>0).astype(np.float32)
                        MasksNew[nm] = ((Masks[nm][:, :, 0] ) > 0).astype(np.float32)
            # if AnnMask.sum()<900:
            #       return self.LoadNext(pos, Hb, Wb)
#----------------------Put loaded data into batch-----------------------------------------------------------------------------------------------------------
            self.BROISemantic[pos][Masks["Vessel"]>0]=1
            for nm in MasksWeighted:
                    self.BMasksSemantic[nm][pos] = MasksNew[nm]
                    self.BMasksWeightedSemantic[nm][pos] = MasksWeighted[nm]
            self.BImgSemantic[pos] = Img

############################################################################################################################################################
# Start load batch of images (multi  thread the reading will occur in background and will will be ready once waitLoad batch as run
    def StartLoadBatchInstance(self): # Start loading instance batch
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        

        #===================Create batch container=================================================
        self.BImgInst = np.zeros([BatchSize, Hb, Wb,3], dtype=np.float32)
        self.BROIInst  = np.zeros([BatchSize, Hb, Wb], dtype=np.float32)
        self.BMasksInst  = []
        self.BMasksWeightedInst =[]
        self.BInsDataInst  = []
        for f in range(BatchSize):
            self.BMasksInst .append({})
            self.BInsDataInst .append({})
            self.BMasksWeightedInst .append({})
        self.thread_listInst = []
        # ====================Start reading data multithreaded===========================================================
        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNextInstance,name="threadInst"+str(pos),args=(pos,Hb,Wb)) # Load single image into batch
            self.thread_listInst.append(th)
            th.start()
###########################################################################################################
#Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatchInstance(self):
            for th in self.thread_listInst:
                 th.join()

########################################################################################################################################################################################
    def LoadBatchInstance(self):# Finish loading instance segmentation training data batch
# Load batch for training (muti threaded  run in parallel with the training proccess)
# return previously  loaded batch and start loading new batch
            self.WaitLoadBatchInstance()
            Imgs=self.BImgInst
            Masks=self.BMasksInst
            InsData=self.BInsDataInst
            ROI=self.BROIInst
            MasksWeighted=self.BMasksWeightedInst
            self.StartLoadBatchInstance()
            return Imgs, Masks,MasksWeighted,ROI,InsData


############################################################################################################################################################
# Start load batch of images (multi  thread the reading will occur in background and will will be ready once waitLoad batch as run
    def StartLoadBatchSemantic(self): # Start loading semantic natch
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb * Wb < self.MaxPixels: break
        BatchSize = np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))

        # ====================Start reading data multithreaded===========================================================

        self.BImgSemantic = np.zeros([BatchSize, Hb, Wb, 3], dtype=np.float32)
        self.BROISemantic = np.zeros([BatchSize, Hb, Wb], dtype=np.float32)
        self.BMasksSemantic = {}
        self.BMasksWeightedSemantic = {}
        for nm in self.ClassToUse:
            self.BMasksSemantic[nm] = np.zeros([BatchSize, Hb, Wb], dtype=np.float32)
            self.BMasksWeightedSemantic[nm] = np.zeros([BatchSize, Hb, Wb], dtype=np.float32)
        self.thread_listSemantic = []
        for pos in range(BatchSize):
            th = threading.Thread(target=self.LoadNextSemantic, name="threadSemantic" + str(pos), args=(pos, Hb, Wb))
            self.thread_listSemantic.append(th)
            th.start()


    ###########################################################################################################
    # Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatchSemantic(self):
        for th in self.thread_listSemantic:
            th.join()


    ########################################################################################################################################################################################
    def LoadBatchSemantic(self): # Finish loading semantic segmentation training data batch
        # Load batch for training (muti threaded  run in parallel with the training proccess)
        # return previously  loaded batch and start loading new batch
        self.WaitLoadBatchSemantic()
        Imgs = self.BImgSemantic
        Masks = self.BMasksSemantic
        ROI = self.BROISemantic
        MasksWeighted = self.BMasksWeightedSemantic
        self.StartLoadBatchSemantic()
        return Imgs, Masks, MasksWeighted, ROI
