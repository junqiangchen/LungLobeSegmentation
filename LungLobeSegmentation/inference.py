from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet.model_vnet3d_multilabel import Vnet3dModuleMultiLabel
from dataprocess.utils import file_name_path
import SimpleITK as sitk


def inference():
    """
        Vnet network segmentation kidney fine segmatation
        :return:
        """
    channel = 1
    numclass = 7
    # step1 init vnet model
    imagex = 144
    imagey = 112
    imagez = 112
    Vnet3d = Vnet3dModuleMultiLabel(imagey, imagex, imagez, channels=channel, numclass=numclass,
                                    costname=("categorical_dice",), inference=True,
                                    model_path="log\VNetwithSize\categorical_focal_loss\model\Vnet3d.pd")
    heart_path = "E:\MedicalData\lung_lobe_Segmentation\\test\Image"
    out_path = "E:\MedicalData\lung_lobe_Segmentation\\test\Predict"
    # step1 get all test image path
    path_list = file_name_path(heart_path, dir=False, file=True)
    # step2 get test image(4 model) and mask
    for subsetindex in range(len(path_list)):
        subset_path = heart_path + "/" + str(path_list[subsetindex])
        heart_src = sitk.ReadImage(subset_path, sitk.sitkInt16)
        heart_array = sitk.GetArrayFromImage(heart_src)
        ys_pd_sitk = Vnet3d.prediction(heart_array / 255.)
        # step3 save output image
        out_mask_image = out_path + "/" + str(path_list[subsetindex])
        sitk.WriteImage(ys_pd_sitk, out_mask_image)


if __name__ == "__main__":
    inference()
