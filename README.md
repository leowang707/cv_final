**CV_final_project**: image Stitching with preprocess by DETR/SETR

# Usage - Object detection
There are no extra compiled components in DETR and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via Docker (recommended) or conda.

## 1. Clone the repository
First, clone the repository locally with ssh:
```
git clone git@github.com:leowang707/cv_final.git
```
If you have no ssh key, you can also clone with https:
```
git clone https://github.com/leowang707/cv_final.git
```

## 2. With Docker (recommended)
Install and build the dockerfile(for first time).
```
cd cv_final/Docker/
```
```
source build.sh
```
Run the docker image.
```
source Docker/docker_run.sh
```

# Usage - Segmentation

We show that it is relatively straightforward to extend DETR to predict segmentation masks. We mainly demonstrate strong panoptic segmentation results.

in the docker
```
source jupyter_notebook.sh
```
in the notebook
Preprocess part
```
run the cell of DETR_panoptic.ipynb
```
or
```
run the cell of SETR_panoptic.ipynb
```
Stitching part(three methods)
```
run the cell of Stitching_case1.ipynb
```
```
run the cell of Stitching_case2.ipynb
```
```
run the cell of Stitching_multi_homo.ipynb
```
