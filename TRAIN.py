# Train for prediction of both semantic maps and instances for the content of a given vessel mask in an image. Should run out of the box with the example set.
#...............................Imports..................................................................
import os
import numpy as np
import FCN_NetModel as NET_FCN # The net Class
import torch
import Reader as LabPicsReader
from scipy.optimize import linear_sum_assignment
import json
import Visuallization as vis
import ClassesGroups
##################################Input paramaters#########################################################################################
#.................................Main Input folder...........................................................................................
ChemLabPicsDir="ExampleTrain/" # Input folder for labpic chemistry training
MedLabPicsDir="ExampleTrain/" # Input folder for labpics medical  training
#......................................Main input parameters..................................................................................


MinSize=250 # min image height width
MaxSize=1200 #  Maxn image height widt
MaxPixels=800*800*3# max number of pixels in a batch reduce to solve out of memoru problems
MinMaskSize=1000 # Min size of vessel mask in pixels, smaller vessels will be ignored
TrainingMode=True # Train or test mode
IgnoreParts=True # Dont train on vessel parts
IgnoreSurfacePhase=False # Don train on materials that are just stuck on the glassware surface (and dont cover volume)
IgnoreScattered=True # Ignore material phases which are scattered in  sparse droplet/particles
MaxBatchSize=6# max images in class
Learning_Rate=1e-4 # initial learning rate
Weight_Decay=1e-5# Weight for the weight decay loss function
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
MAX_ITERATION = int(100000010) # Max  number of training iteration
#********************************Classes to use in training the semantic segmentation the net will only produce maps for this clases************************************************************************************************************
MaterialClasses=ClassesGroups.MaterialClass
ClassToUse=ClassesGroups.VesselContentClasses
#****************************************************************************************************************************************************

if not os.path.exists(TrainedModelWeightDir):
    os.mkdir(TrainedModelWeightDir)
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""

#=========================Load net weights and parameters from previous runs if exist====================================================================================================================
InitStep=1
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
    Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"):
    Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))

#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(ClassList=ClassToUse) # Create neural net


if Trained_model_path!="": # Optional initiate full net by loading a previosly trained net weights if exist
    Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda() # Train on cuda
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer

#----------------------------------------Create readers for data set--------------------------------------------------------------------------------------------------------------

ChemReader=LabPicsReader.Reader(ChemLabPicsDir,MaxBatchSize,MinSize,MaxSize,MaxPixels,MinMaskSize, TrainingMode,IgnoreParts,IgnoreSurfacePhase,IgnoreScattered,ClassToUse=ClassToUse)
MedReader=LabPicsReader.Reader(MedLabPicsDir,MaxBatchSize,MinSize,MaxSize,MaxPixels,MinMaskSize, TrainingMode,IgnoreParts,IgnoreSurfacePhase,IgnoreScattered,ClassToUse=ClassToUse)

#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
#-------------------statitics  paramters for tracking loss--------------------------------------------------------------------------------
PrevAvgInsLoss=0 # Previously average instance loss
PrevAvgSemLoss=0 # previously average semantic loss
AVGInsLoss=0  #  average instance loss
AVGSemLoss=0  # average semantic loss
AVGCatLoss={} # Average loss per category  for semantic segmentation
for nm in ClassToUse:
    AVGCatLoss[nm]=0
#..............Start Training loop: Main Training....................................................................
print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop
    fr = 1 / np.min([itr - InitStep + 1, 2000])
    print(itr)
    Mode=""
#-----------------------------Read data----------------------------------------------------------------
    if np.random.rand() < 0.4: # Read instance data
            Mode = "Instance"
            if np.random.rand()<0.62:
                Imgs, GTMasks,WeightMaps, ROI, InsData = ChemReader.LoadBatchInstance()# Read from labpic chemistry
            else:
                Imgs, GTMasks, WeightMaps, ROI, InsData = MedReader.LoadBatchInstance()# Read from labpic medical
    else: # Read semantic data
            Mode = "Semantic"
            if np.random.rand()<0.62:
                Imgs, GTMasks,WeightMaps, ROI = ChemReader.LoadBatchSemantic() # Read from labpic chemistry
            else:
                Imgs, GTMasks, WeightMaps, ROI = MedReader.LoadBatchSemantic() # Read from labpic medical
    #----------------Display readed data----------------------
    # for i in range(Imgs.shape[0]):
    #     I = Imgs[i].copy()
    #     I1 = I.copy()
    #     I1[:, :, 0][ROI[i] > 0] = 0
    #     I1[:, :, 1][ROI[i] > 0] = 0
    #     vis.show(np.concatenate([I, I1], axis=1), "Vessel  ")
    #     for nm in GTMasks[i]:
    #         if nm == 'Vessel': continue
    #         I2 = I.copy()
    #         I2[:, :, 0][GTMasks[i][nm] > 0] = 0
    #         I2[:, :, 1][GTMasks[i][nm] > 0] = 0
    #       #  print(InsData[i][nm])
    #       #  vis.show(np.concatenate([I, I1, I2, vis.GreyScaleToRGB(WeightMaps[i][nm] * 255)], axis=1),str(InsData[i][nm]))\
    #         vis.show(np.concatenate([I, I1, I2, vis.GreyScaleToRGB(WeightMaps[i][nm] * 255)], axis=1),nm+str(np.unique(WeightMaps[i][nm])))
    #-----------------------------------------------------------------------------------------------------------------
    # for i in range(Imgs.shape[0]):
    #     I = Imgs[i].copy()
    #     I1 = I.copy()
    #     I1[:, :, 0][ROI[i] > 0] = 0
    #     I1[:, :, 1][ROI[i] > 0] = 0
    #     vis.show(np.concatenate([I, I1], axis=1), "Vessel  ")
    #     for nm in GTMasks:
    #         if GTMasks[nm][i].sum()==0: continue
    #         I2 = I.copy()
    #         I2[:, :, 0][GTMasks[nm][i] > 0] = 0
    #         I2[:, :, 1][GTMasks[nm][i] > 0] = 0
    #       #  print(InsData[i][nm])
    #       #  vis.show(np.concatenate([I, I1, I2, vis.GreyScaleToRGB(WeightMaps[i][nm] * 255)], axis=1),str(InsData[i][nm]))\
    #         vis.show(np.concatenate([I, I1, I2, vis.GreyScaleToRGB(WeightMaps[nm][i] * 255)], axis=1),nm+str(np.unique(WeightMaps[nm][i])))
    #     continue
    #----------------------------------------------
    #print("RUN PREDICITION")
#-------------------------------------Run Prediction--------------------------------------------------------------------------------
    ProbInst, LbInst, ProbSemantic, LbSemantic = Net.forward(Images=Imgs,ROI=ROI,PredictSemantic=(Mode == "Semantic"),PredictInstance = (Mode == "Instance")) # Run net inference and get prediction
    Net.zero_grad()
    Loss=0
    batchSize = Imgs.shape[0]
#******************Instance Segmentation loss find best matching GT /Predicted) segments (hungarian matching)**************************************************************************
    if (Mode == "Instance"):
        for iii in range(len(GTMasks)): # Note the matching  and loss is done now for every image of the batch indepenently. This is not very effiecnt p
            LbSum = {}
            GTMasksPT={}
            WeightMapsPT={}
            Cost = np.zeros([len(LbInst), len(GTMasks[iii])], dtype=np.float32) # Create correlation matrix of every IOU between every GT segment and predicted segment
            for ff,nm in enumerate(GTMasks[iii]):
                GTMasksPT[ff]=torch.autograd.Variable(torch.from_numpy((GTMasks[iii][nm]).astype(np.float32)).cuda(), requires_grad=False)
                WeightMapsPT[ff]=torch.autograd.Variable(torch.from_numpy((WeightMaps[iii][nm]).astype(np.float32)).cuda(), requires_grad=False)
                GTMasksSum=GTMasksPT[ff].sum()
                for ip in range(len(LbInst)):
                    if not ip in LbSum:
                        LbSum[ip]=LbInst[ip][iii].sum()
                    inter=(LbInst[ip][iii]*GTMasksPT[ff]).sum()
                    iou=inter/(GTMasksSum+LbSum[ip]-inter+0.0000001) # Calculate IOU between predicted and GT segment
                    Cost[ip,ff]=iou.data.cpu().numpy()
    #--------------------------Find match and calculate loss-------------------------------------------------------------
            row_ind, col_ind = linear_sum_assignment(-Cost) # Hungarian matching find the best matching prediction to GT segment, based on IOU matching
            for i in range(len(row_ind)): # Caclulate crossentropy  loss between  matching predicted and GT masks
                Loss -= torch.mean(GTMasksPT[col_ind[i]]* torch.log(ProbInst[row_ind[i]][iii][1]+0.000001)*WeightMapsPT[col_ind[i]])
                Loss -= torch.mean((1-GTMasksPT[col_ind[i]])* torch.log(ProbInst[row_ind[i]][iii][0]+0.000001))#
            for i in range(len(ProbInst)): # For unmatched set GT to empty segment
                if i not in (row_ind):
                    Loss -= torch.mean(torch.log(ProbInst[i][iii][0] + 0.000001))
        Loss /= batchSize
        AVGInsLoss = (1 - fr) * AVGInsLoss + fr * Loss.data.cpu().numpy() # Add to average loss for statitics
#*********************************Calculate semantic loss******************************************************************************************
    if Mode == "Semantic":
        Loss = 0
        LossByCat = {}
        lossCount = 0
        #  print("lll===="+str(len(list(OutLbDict))))
        #  print("-2-")
        for nm in ProbSemantic: # Go over all classes and match semantic maps
            lossCount += 1
            GT = torch.autograd.Variable(torch.from_numpy(GTMasks[nm].astype(np.float32)).cuda(), requires_grad=False)
            WeightMapsPT = torch.autograd.Variable(torch.from_numpy(WeightMaps[nm].astype(np.float32)).cuda(), requires_grad=False)
            LossByCat[nm] = -torch.mean((GT * torch.log(ProbSemantic[nm][:, 1, :, :] + 0.0000001) * WeightMapsPT + (1 - GT) * torch.log(ProbSemantic[nm][:, 0, :, :] + 0.0000001)))
            Loss += LossByCat[nm]
            AVGCatLoss[nm] = AVGCatLoss[nm] * (1 - fr) + fr * float(LossByCat[nm].data.cpu().numpy())  # Intiate runing average loss
            AVGSemLoss = (1 - fr) * AVGSemLoss + fr * Loss.data.cpu().numpy() # average semantic segmentation loss for statitics
#-------------------------------------------------------------------------------------------------------------------

    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight

####################################Saving and displaying###########################################################################

# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 1000 == 0:# and itr>0: #Save model weight once every 1000 steps temporary
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 60000 == 0 and itr>0: #Save model weight once every 60k steps permenant
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display train loss

        txt="\n"+str(itr)+"\t Semantic Loss "+str(AVGSemLoss)+"\t Ins Loss "+str(AVGInsLoss)+"\t Learning Rate "+str(Learning_Rate) +"\n"
        for nm in ClassToUse:
            txt+=nm+") "+str(AVGCatLoss[nm])+"  "
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
#----------------Update learning rate fractal manner-------------------------------------------------------------------------------
    if itr%10000==0:
        if PrevAvgInsLoss*0.95<AVGInsLoss or PrevAvgSemLoss*0.95<AVGSemLoss: # if average loss as not decrease in the last 10k step reduce learning rate
            Learning_Rate*=0.9 # Reduce learing rate
            if Learning_Rate<=4e-7: # If learning rate is to small increase it back up
                Learning_Rate=5e-6
            print("Learning Rate="+str(Learning_Rate))
            print("======================================================================================================================")
            optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer with new loss
            torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
        PrevAvgInsLoss=AVGInsLoss+0.0000000001
        PrevAvgSemLoss= AVGSemLoss+0.0000001


