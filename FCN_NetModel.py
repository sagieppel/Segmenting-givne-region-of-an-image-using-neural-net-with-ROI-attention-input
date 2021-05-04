#  Generate FCN  with  vessel mask as a additional input a

import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Visuallization as vis
######################################################################################################################333
class Net(nn.Module):
    def __init__(self, ClassList):
            # --------------Build layers for standart FCN with only image as input------------------------------------------------------
            super(Net, self).__init__()
            # ---------------Load pretrained  Resnet 50 encoder----------------------------------------------------------
           # self.Encoder = models.resnext101_32x8d(pretrained=True)
            #self.Encoder = models.resnet50(pretrained=True)
            self.Encoder = models.resnext50_32x4d(pretrained=True)
            #self.Encoder = models.resnext101_32x8(pretrained=True)
            # --------------- -------------------------------------------------------------------------
            self.PSPScales = [1, 1 / 4, 1 / 8, 1 / 16] # scalesPyramid Scene Parsing PSP layer
            self.ASPPScales=[1, 4, 12, 24] # scales ASPP deep lab 3 net
        # --------------------Proccess ROI/vessel mask ----------------------------------------------------------------------

            self.AttentionLayers = nn.ModuleList()
            self.ROIEncoder = nn.Conv2d(1, 64, stride=2, kernel_size=3, padding=1, bias=True)
            self.ROIEncoder.bias.data = torch.ones(self.ROIEncoder.bias.data.shape)
            self.ROIEncoder.weight.data = torch.zeros(self.ROIEncoder.weight.data.shape)
    #---------------------------------PSP layer----------------------------------------------------------------------------------------
            self.PSPLayers = nn.ModuleList()  # [] # Layers for decoder

            for Ps in self.PSPScales:
                self.PSPLayers.append(nn.Sequential(
                    nn.Conv2d(2048, 512, stride=1, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(512),nn.ReLU()))
#----------------------------------------ASPP  deeplab layers-----------------------------------------------------------------------
            self.ASPPLayers = nn.ModuleList()
            for scale in self.ASPPScales:
                    self.ASPPLayers.append(nn.Sequential(
                    nn.Conv2d(2048, 512, stride=1, kernel_size=3,  padding = (scale, scale), dilation = (scale, scale), bias=False),nn.BatchNorm2d(512),nn.ReLU()))

#)
#-------------------------------------------------------------------------------------------------------------------
            self.SqueezeLayers = nn.Sequential(
                nn.Conv2d(2048, 512, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()#,
                # nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm2d(512),
                # nn.ReLU()
            )
            # ------------------Skip conncetion layers for upsampling-----------------------------------------------------------------------------
            self.SkipConnections = nn.ModuleList()
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(256, 128, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()))
            # ------------------Upsampling +Squeeze layers for(concat of upsample+skip conncection layers)-----------------------------------------------------------------------------
            self.SqueezeUpsample = nn.ModuleList()
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 512, 256, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 128, 256, stride=1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))


            # ----------------Final prediction Instance------------------------------------------------------------------------------------------

            self.OutLayersListInst =nn.ModuleList()
           # self.OutLayersDict={}
            for f in range(10): # Create binary  prediction for each instance
                    self.OutLayersListInst.append(nn.Conv2d(256, 2, stride=1, kernel_size=3, padding=1, bias=False))
            # ----------------Final prediction Semantic------------------------------------------------------------------------------------------
            self.OutLayersListSemantic = nn.ModuleList()
            self.OutLayersDictSemantic={}
            for nm in ClassList: # Create binary  prediction for each semantic map
                        self.OutLayersDictSemantic[nm]=nn.Conv2d(256, 2, stride=1, kernel_size=3, padding=1, bias=False)
                        self.OutLayersListSemantic.append(self.OutLayersDictSemantic[nm])

##########################################################################################################################################################
    def forward(self, Images, ROI, UseGPU=True, TrainMode=True,PredictSemantic=True,PredictInstance=True, FreezeBatchNorm_EvalON=False):

               # ----------------------Convert image to pytorch and normalize values-----------------------------------------------------------------
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]
                if TrainMode == True:
                   tp = torch.FloatTensor
                else:
                   tp = torch.half
                   #      self.eval()
                   self.half()
                if FreezeBatchNorm_EvalON: self.eval()

                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(float)), requires_grad=False).transpose(2,3).transpose(1, 2).type(tp)

               # -------------------Convert ROI mask into pytorch format----------------------------------------------------------------
                ROImap = torch.autograd.Variable(torch.from_numpy(ROI.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(tp)

               # ---------------Convert to cuda gpu-------------------------------------------------------------------------------------------------------------------
                if UseGPU:
                   ROImap = ROImap.cuda()
                   InpImages = InpImages.cuda()
                   self.cuda()
                else:
                   ROImap = ROImap.cpu().float()
                   InpImages = InpImages.cpu().float()
                   self.cpu().float()
#----------------Normalize image values-----------------------------------------------------------------------------------------------------------
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # normalize image values
                x=InpImages
#--------------------------------------------------------------------------------------------------------------------------
                SkipConFeatures=[] # Store features map of layers used for skip connection
#---------------Run Encoder first layer-----------------------------------------------------------------------------------------------------
                x = self.Encoder.conv1(x)
                x = self.Encoder.bn1(x)
#------------------------Convery ROI/Vessel mask  map into attention layer and merge with image feature mask-----------------------------------------------------------
                r = self.ROIEncoder(ROImap) # Generate attention map from ROI mask
                x = x*r # Merge feature mask and attention maps
#-------------------------Run remaining encoder layer------------------------------------------------------------------------------------------
                x = self.Encoder.relu(x)
                x = self.Encoder.maxpool(x)
                x = self.Encoder.layer1(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer2(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer3(x)
                SkipConFeatures.append(x)
                EncoderMap = self.Encoder.layer4(x)
#------------------Run psp  Layers (using aspp instead)----------------------------------------------------------------------------------------------
                # PSPSize=(EncoderMap.shape[2],EncoderMap.shape[3]) # Size of the original features map
                # PSPFeatures=[] # Results of various of scaled procceessing
                # for i,PSPLayer in enumerate(self.PSPLayers): # run PSP layers scale features map to various of sizes apply convolution and concat the results
                #       NewSize=(np.array(PSPSize)*self.PSPScales[i]).astype(np.int)
                #       if NewSize[0] < 1: NewSize[0] = 1
                #       if NewSize[1] < 1: NewSize[1] = 1
                #
                #       # print(str(i)+")"+str(NewSize))
                #       y = nn.functional.interpolate(EncoderMap, tuple(NewSize), mode='bilinear',align_corners=False)
                #       #print(y.shape)
                #       y = PSPLayer(y)
                #       y = nn.functional.interpolate(y, PSPSize, mode='bilinear',align_corners=False)
                #       PSPFeatures.append(y)
                # x=torch.cat(PSPFeatures,dim=1)
                # x=self.SqueezeLayers(x)

#---------------------------------ASPP Layers--------------------------------------------------------------------------------
                ASPPFeatures = []  # Results of various of scaled procceessing
                for ASPPLayer in self.ASPPLayers:
                    y = ASPPLayer( EncoderMap )
                    ASPPFeatures.append(y)
                x = torch.cat(ASPPFeatures, dim=1)
                x = self.SqueezeLayers(x)
#----------------------------Upsample features map  and combine with layers from encoder using skip  connection-----------------------------------------------------------------------------------------------------------
                for i in range(len(self.SkipConnections)):
                  sp=(SkipConFeatures[-1-i].shape[2],SkipConFeatures[-1-i].shape[3])
                  x=nn.functional.interpolate(x,size=sp,mode='bilinear',align_corners=False) #Resize
                  x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1-i]),x), dim=1)
                  x = self.SqueezeUpsample[i](x)
#********************************************************************************************************
               # ---------------------------------Final prediction-------------------------------------------------------------------------------

                self.OutProbInst = []
                self.OutLbInst = []
                if PredictInstance: # predict instances
                    for layer in self.OutLayersListInst:
                        # print(nm)
                        l = layer(x)
                        l = nn.functional.interpolate(l, size=InpImages.shape[2:4], mode='bilinear',align_corners=False)  # Resize to original image size
                        Prob = F.softmax(l, dim=1)  # Calculate class probability per pixel
                        tt, Labels = l.max(1)  # Find label per pixel
                        self.OutProbInst.append(Prob)
                        self.OutLbInst.append(Labels)
               # ********************************************************************************************************
                self.OutProbSemantic = {}
                self.OutLbSemantic = {}
                if PredictSemantic: # Predict semanitc maps
                    for nm in self.OutLayersDictSemantic:
                        l=self.OutLayersDictSemantic[nm](x)
                        l = nn.functional.interpolate(l, size=InpImages.shape[2:4], mode='bilinear',align_corners=False)  # Resize to original image size
                        Prob = F.softmax(l, dim=1)  # Calculate class probability per pixel
                        tt, Labels = l.max(1)  # Find label per pixel
                        self.OutProbSemantic[nm]=Prob
                        self.OutLbSemantic[nm]=Labels
                return self.OutProbInst, self.OutLbInst,self.OutProbSemantic, self.OutLbSemantic









