# Apply Inference to LabPics test, output format match LabPics evaluator input. Run out of the box with the example datas
#...............................Imports..................................................................
import os
import numpy as np
import FCN_NetModel as NET_FCN # The net Class
import torch
import Reader as LabPicsReader
from scipy.optimize import linear_sum_assignment
import json
import Visuallization as vis
import cv2
import shutil
import json
import InferenceSingle as Infer
import ClassesGroups
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................

OutDir="OutTestPrediction/"  # Output folder were prediction will be saved (Labpics format)
InputDir="ExampleTrain/" #  Input for net prediction (Lab pics test set format)
Trained_model_path="logs/Defult.torch"

UseGPU=True
if os.path.exists(OutDir): shutil.rmtree(OutDir)
os.mkdir(OutDir)


#********************************************************************************************************************************************
ClassToUse=ClassesGroups.VesselContentClasses
Net=NET_FCN.Net(ClassList=ClassToUse) # Create net and load pretrained
Net.load_state_dict(torch.load(Trained_model_path))
if UseGPU: Net=Net.cuda().eval()
#...................................Start Training loop: Main Training....................................................................
for dr in os.listdir(InputDir):
    print(dr)
    #............................Create output folders.............................................
    MainOutDir = OutDir + "/" + dr
    os.mkdir(MainOutDir)
    OutMaterialInstDir =  MainOutDir+"/ContentInstances/"
    os.mkdir(OutMaterialInstDir)
    OutMaterialSemanticDir = MainOutDir + "/ContentSemantic/"
    os.mkdir(OutMaterialSemanticDir)



    MainDir=InputDir + "/" + dr
    ImagePath = MainDir + "/Image.jpg"  # Load Image
    VesMaskDir=MainDir+"/Vessels//"
    shutil.copytree(VesMaskDir,MainOutDir+"/Vessels//")
    shutil.copy(ImagePath,MainOutDir+ "/Image.jpg")
    #......................................Go over all vessels and predict content...........................................................
    for fl in os.listdir(VesMaskDir): #
        OutInstSubDir = OutMaterialInstDir + "/" + fl.replace(".png", "")
        OutSemSubDir = OutMaterialSemanticDir + "/" + fl.replace(".png", "")

        VesselMaskPath = VesMaskDir+"/"+fl
        #.............................................Run prediction on a single vessel and save results
        Infer.InferSingleVessel(Net,ImagePath,VesselMaskPath,OutDirInst=OutInstSubDir,OutDirSemantic=OutSemSubDir,ClassesToUse=ClassToUse, UseGPU=UseGPU)








