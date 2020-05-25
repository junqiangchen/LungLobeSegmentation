from __future__ import print_function, division
import numpy as np
import SimpleITK as sitk
import os
from dataprocess.utils import file_name_path, resize_image_itkwithsize, ConvertitkTrunctedValue, getRangImageRange


def preparesampling3dtraindata(heart_imagepath, heart_labelpath, trainImage, trainMask, shape=(96, 96, 96)):
    mask_path_list = file_name_path(heart_labelpath, False, True)
    image_path_list = file_name_path(heart_imagepath, False, True)
    newSize = shape
    for subsetindex in range(len(image_path_list)):
        # step1 load mask image ,then resize to new Spacing
        mask_path = heart_labelpath + "/" + str(mask_path_list[subsetindex])
        seg = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        z_min, z_max = getRangImageRange(seg_array, 0)
        y_min, y_max = getRangImageRange(seg_array, 1)
        x_min, x_max = getRangImageRange(seg_array, 2)
        roi_seg = sitk.GetImageFromArray(seg_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1])
        roi_seg.SetOrigin(seg.GetOrigin())
        roi_seg.SetSpacing(seg.GetSpacing())
        roi_seg.SetDirection(seg.GetDirection())
        _, roi_seg = resize_image_itkwithsize(roi_seg, newSize, roi_seg.GetSize(), sitk.sitkNearestNeighbor)
        segimg = sitk.GetArrayFromImage(roi_seg)
        # step2 load src image with window center and window level,then resize to new Spacing
        file_image = heart_imagepath + "/" + str(image_path_list[subsetindex])
        src = sitk.ReadImage(file_image, sitk.sitkInt16)
        src_array = sitk.GetArrayFromImage(src)
        roi_src = sitk.GetImageFromArray(src_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1])
        roi_src.SetOrigin(src.GetOrigin())
        roi_src.SetSpacing(src.GetSpacing())
        roi_src.SetDirection(src.GetDirection())
        _, roi_src = resize_image_itkwithsize(roi_src, newSize, roi_src.GetSize(), sitk.sitkLinear)
        roi_src = ConvertitkTrunctedValue(roi_src, 600, -1000, True)
        srcimg = sitk.GetArrayFromImage(roi_src)
        # step 3 get subimages and submasks
        if not os.path.exists(trainImage):
            os.makedirs(trainImage)
        if not os.path.exists(trainMask):
            os.makedirs(trainMask)
        filepath1 = trainImage + "\\" + str(subsetindex) + ".npy"
        filepath = trainMask + "\\" + str(subsetindex) + ".npy"
        np.save(filepath1, srcimg)
        np.save(filepath, segimg)


def preparetraindata():
    """
    :return:
    """
    heart_path = "E:\MedicalData\lung_lobe_Segmentation\\train"
    image_name = "Image"
    mask_name = "Mask"
    heart_imagepath = heart_path + "/" + image_name
    heart_labelpath = heart_path + "/" + mask_name
    trainImage = "data\Image"
    trainMask = "data\Mask"
    preparesampling3dtraindata(heart_imagepath, heart_labelpath, trainImage, trainMask, (144, 112, 112))


if __name__ == "__main__":
    preparetraindata()
