# FCOS_POSE
Codes for my paper *"STUDENT DANGEROUS BEHAVIOR DETECTION IN SCHOOL"*. Currently, we have only tested the effectiveness of our method on the personal collected danger behavior dataset. You can build a similar dataset by yourself, and transfer our `FCOS_POSE` framework based on FCOS (object detection) and auxiliary keypoints (pose estimation) to your human behavior detection task.

![example1](./materials/network_architecture.png)

## Installation
**Note:** Our FCOS_POSE is mostly based on the original object detection method [FCOS](https://github.com/tianzhi0549/FCOS). You can follow it to set the environment. Our pose estimation backbone is the [StackedHourglasses](https://github.com/princeton-vl/pytorch_stacked_hourglass).

**Environment:** Anaconda, Python3.8, PyTorch1.10.0(CUDA11.2)

``` bash
$ git clone https://github.com/hnuzhy/fcos_pose.git
$ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Codes are only evaluated on GTX3090+CUDA11.2+PyTorch1.10.0. You can follow the same config if needed
# [method 1][directly install from the official website][may slow]
$ pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/cu111/torch_stable.html
  
# [method 2]download from the official website and install offline][faster]
$ wget https://download.pytorch.org/whl/cu111/torch-1.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl
$ wget https://download.pytorch.org/whl/cu111/torchvision-0.11.1%2Bcu111-cp38-cp38-linux_x86_64.whl
$ wget https://download.pytorch.org/whl/cu111/torchaudio-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl
$ pip3 install torch*.whl

# install the lib with symbolic links
$ python setup.py build develop --no-deps
```

## Dataset Preparing

**Note:** We are sorry that due to privacy issues, our personal dangerous behavior dataset cannot be released for public use. We can only provide an overview of the data set production process.

* **Step 1:** Collect videos containing dangerous behaviors (e.g., `fight`, `tumble`, and `squat`). Sample frames every N (e.g., 3) seconds. Then, annotate dangerous behaviors with bounding boxes. You can use the popular online annotation tool [CVAT](https://cvat.org/).
* **Step 2:** After annotation (about 10K images totally), export your dataset in the `PascalVOC` format (`JPEGImages` folder contains selected frames, `Annotations` folder contains corresponding labels in XML format).
* **Step 3:** Convert the annotations from `PascalVOC` XML style to `COCO` JSON style. We recommend you to refer the scripts in [voc2coco](https://github.com/yukkyo/voc2coco). Support that you finally get two (train and val) json annotation files `coco_dangerdet_train.json` and `coco_dangerdet_val.json`.
* **Step 4:** Generate keypoints for all persons appearing in frames under the `JPEGImages` folder. We use the robust bottom-up method [OpenPifPaf](https://github.com/vita-epfl/openpifpaf) to detect all auxiliary skeletons. Because they are auxiliary labels, the imprecision of keypoints will not affect the normal execution of training. Support that you finally get all keypoints json files under the `Keypoints_JSON` folder.
* **Step 5:** Update the json annotation files in **Step 3** with adding keypoints information. We provide an example script [voc2cocoWpose.py](./tools/[voc2cocoWpose.py). Finally, you will get two new json annotation files `coco_dangerdet_Wpose_train.json` and `coco_dangerdet_Wpose_val.json`.


## Training and Testing

* **Yamls**

* **Training**

* **Testing**

## References
* [FCOS: Fully Convolutional One-Stage Object Detection](https://github.com/tianzhi0549/FCOS)
* [Stacked Hourglass Networks in Pytorch](https://github.com/princeton-vl/pytorch_stacked_hourglass)
