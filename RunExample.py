# Run example on single image and single mask (should run out of the box for a the example image and mask)
#...............................Imports..................................................................
import os
import FCN_NetModel as NET_FCN # The net Class
import torch
import shutil
import InferenceSingle as Infer
import ClassesGroups
##################################Input paramaters#########################################################################################


TestImage="ExamplePredict/Image.jpg" # Input image
TestMask="ExamplePredict/Mask.png" # input vessel mask

OutDir="OutPrediction/"  # Output prediction dir
Trained_model_path="logs/Defult.torch" # Pretrain model path


#########################Create output folders#####################################################
if os.path.exists(OutDir): shutil.rmtree(OutDir)
os.mkdir(OutDir)


UseGPU=False # use GPU

OutDirInst=OutDir+"/Instance/"
OutDirSemantic=OutDir+"/Semantic/"
OutDirInstDiplay=OutDir+"/InstanceOverlay/"
OutDirSemanticDisplay=OutDir+"/SemanticOverlay/"
#**************************Load model and create net******************************************************************************************************************
ClassToUse=ClassesGroups.VesselContentClasses
Net=NET_FCN.Net(ClassList=ClassToUse) # Create net and load pretrained
Net.load_state_dict(torch.load(Trained_model_path))
if UseGPU: Net=Net.cuda().eval()
#...................................Run prediction and save results....................................................................
Infer.InferSingleVessel(Net,TestImage,TestMask,OutDirInst,OutDirSemantic,OutDirInstDiplay,OutDirSemanticDisplay,ClassesToUse=ClassToUse,UseGPU=UseGPU)





