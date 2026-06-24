# Dataset Download Instructions

To train and evaluate the HM-RL-FusionMamba framework, prepare the paired multi-modal datasets according to the following instructions.

---

## 1. LLVIP Dataset (Visible-Infrared Paired Dataset)
LLVIP is a high-quality dataset containing visible and infrared image pairs of pedestrians, mostly captured at night.

1. **Download Link**: Visit the official repository at [LLVIP Github](https://github.com/bupt-ai-cz/LLVIP).
2. **Extraction**:
   - Download the infrared and visible image archives.
   - Extract them and place the images in the following directory layout:
     ```
     datasets/LLVIP/infrared/ (e.g., 010001.png, 010002.png)
     datasets/LLVIP/visible/  (e.g., 010001.png, 010002.png)
     ```
   - Ensure the filenames in `infrared` and `visible` match exactly (e.g. `010001.png` is present in both directories).

---

## 2. FLIR ADAS Dataset (Thermal ADAS Dataset)
FLIR provides an annotated thermal dataset for advanced driver assistance systems (ADAS).

1. **Download Link**: Register and download from [FLIR ADAS Dataset Form](https://www.flir.com/oem/adas/adas-dataset-form/) or search on Kaggle for pre-packaged FLIR ADAS datasets.
2. **Extraction**:
   - Locate the paired thermal (infrared) and visible RGB camera files.
   - Place them in the following directories:
     ```
     datasets/FLIR/infrared/
     datasets/FLIR/visible/
     ```
   - Note: Some versions of the FLIR dataset contain unaligned visible/thermal pairs. Use only the aligned/paired subsets or map them by matching their names or annotations.

---

## 3. M3FD Dataset (Optional)
M3FD contains diverse visible-infrared pairs in outdoor and driving scenes.

1. **Download Link**: Follow instructions in the [M3FD Github Repository](https://github.com/JinyuanLiu-COPRE/M3FD).
2. **Extraction**:
   - Align the infrared and visible subdirectories.
   - Place them under `datasets/M3FD/infrared` and `datasets/M3FD/visible` if you want to extend evaluations to M3FD.
