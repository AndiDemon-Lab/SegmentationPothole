# Road Damage Segmentation Using U-Net

This repository contains the implementation of **road damage semantic segmentation** using the **U-Net** architecture on a manually collected road damage dataset from **Purwokerto, Indonesia**.

## Features

- Semantic road damage segmentation using U-Net
- Manually annotated dataset (LabelMe polygon)
- Benchmark with Mask R-CNN and DeepLabV3
- Metadata included

---

## Dataset

The complete dataset is available on Google Drive.

**Download Dataset**

**https://drive.google.com/drive/folders/1us0VOCkAL5taUgJkPRzBd55NI8aVnY4q?usp=sharing**

The dataset includes:

- Road images (.jpg)
- Segmentation masks (.png)
- LabelMe annotations (.json)
- Metadata (.csv)

---

## Folder Structure

```text
SegmentationPothole
│
├── images/
│   ├── Pothole_1.jpg
├── masks/
│   ├── Pothole_1.png
├── dataset_split/
│   ├── train.txt
│   ├── test.txt
│   ├── trainmaskrcnn.txt
│   ├── testmaskrcnn.txt
├── annotation/
│   └── dataset_meta.json
├── deeplabv/
├── maskrcnn/
├── unet/
├── result_train/
├── resultmaskrcnn/
├── resultdeeplabv/

```

---

##  Citation

If you use this repository or dataset in your research, please cite the corresponding paper.

---

##  Author

**Naufal El Kamil Aditya Pratama Rahaman**  
Informatics Engineering Department  
Directorate of Purwokerto Campus  
Telkom University
