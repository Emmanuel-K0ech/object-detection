# Object Detection Project

## Overview

This repository contains code and resources for creating a custom object detection model using TensorFlow. The project walks through the entire workflow from annotating images to generating TensorFlow Lite models suitable for mobile deployment.

## Table of Contents

- [Object Detection Project](#object-detection-project)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Work Accomplished](#work-accomplished)
    - [Generating TF Record](#generating-tf-record)
    - [Training the Model](#training-the-model)
    - [Generating the TFLite Folder](#generating-the-tflite-folder)
  - [Challenges Experienced](#challenges-experienced)
  - [Fixing the Challenges](#fixing-the-challenges)
  - [Errors Experienced](#errors-experienced)
  - [References](#references)

## Work Accomplished

### Generating TF Record

The following steps were followed to generate TF Records:

1. **Cloning the labelImg repository** from its [GitHub repository](https://github.com/tzutalin/labelImg).
2. **Setting up the environment:**
   - Download Anaconda and open it as administrator.
   - Create and activate a new virtual environment:
     ```bash
     conda create -n 'name_of_virtual_environment'
     conda activate 'name_of_virtual_environment'
     ```
3. **Installing required packages** in the Anaconda terminal:
   ```bash
   conda install pyqt=5
   cd labelImg
   pyrcc5 -o libs/resources.py resources.qrc
   pip install lxml
   pip install pyqt5
   python labelImg.py
   ```
   The annotation tools will pop up on the screen.
4. **Annotating the objects** in the images using the annotation tools.
5. **Generating CSV files** from XML files using `xml_to_csv.py` and `generate_tfrecord.py`.
6. **Organizing dataset**: Move 10% of the images and their corresponding CSV files to the test folder and the rest to the train folder.
   ```bash
   python xml_to_csv.py
   ```
7. **Configuring the `generate_tfrecord.py` file**:
   - Modify the labels and paths in the code.
   - Run the following commands:
     ```bash
     python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/
     python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/
     ```

### Training the Model

8. **Installing requirements** for training the custom model:
   ```bash
   pip install TensorFlow==1.15 lxml pillow matplotlib jupyter contextlib2 cython tf_slim
   ```
9. **Cloning the TensorFlow models repository** from [this link](https://github.com/tensorflow/models).
10. **Installing `protoc` version 3.4**:
    - Extract the zip and move `protoc.exe` into the ‘research’ folder.
    - Compile with:
      ```bash
      protoc object_detection/protos/*.proto --python_out=.
      ```
11. **Building and installing the object detection scripts**:
    ```bash
    python setup.py build
    python setup.py install
    ```
12. **Downloading the COCO trained model** and config file from [this link](https://github.com/tensorflow/models/tree/master/research/object_detection/g3doc/tflite/models).
13. **Editing the config file** to configure paths and classes.
14. **Creating a PBtxt file** for the training folder.
15. **Setting the PYTHONPATH**:
    ```bash
    set PYTHONPATH=$PYTHONPATH:pwd:pwd/slim
    ```
16. **Training the model**:
    ```bash
    python train.py --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_coco.config --logtostderr
    ```
    Model files and checkpoints will be generated during training.

### Generating the TFLite Folder

17. **Creating a TensorFlow frozen graph** with compatible ops:
    ```bash
    python export_tflite_ssd_graph.py --pipeline_config_path=training/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix=training/model.ckpt-50233 --output_directory=tflite --add_postprocessing_op=true
    ```
18. **Converting to TensorFlow Lite format**:
    ```bash
    tflite_convert --graph_def_file=tflite/tflite_graph.pb --output_file=tflite/detect.tflite --output_format=TFLITE --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=127 --change_concat_input_ranges=false --allow_custom_ops
    ```

## Challenges Experienced

- Incompatible requirement versions (e.g., TensorFlow 1.15 with Python 3.8/3.9).
- Misguided directories and unconfirmed paths in code.
- Broken packages (e.g., matplotlib, pylint).
- Outdated or incompatible command syntax.

## Fixing the Challenges

- Uninstalling and reinstalling packages to recover from broken installations.
- Double-checking code and paths.
- Using specific versions (e.g., TensorFlow 1.15, Python 3.7).
- Forcing installations using:
  ```bash
  conda install -c conda-forge 'name_of_app'
  ```

## Errors Experienced

- Installing TensorFlow version 1.15:
  ```bash
  pip install TensorFlow==1.15
  ```
- Installing the `tf_slim` module:
  ```bash
  pip install tf_slim
  pip uninstall tf_slim
  ```
- Installing `numpy` version 1.19.5:
  ```bash
  pip install numpy==1.19.5
  ```

## References

- Ben, G. (2020, July 16). [How to create a custom Object Detector with TensorFlow 2020](https://www.youtube.com/watch?v=C5-SEZ_IvaM&t=1068s).
- Ben, G. (2020, July 22). [How to Train a Custom Model for Object Detection (Local and Google Colab!)](https://www.youtube.com/watch?v=_gGI91BmIdk&t=610s).
- Toure, N. (2019, June 8). [Convert a TensorFlow frozen graph to a TensorFlow lite (tflite) file (Part 3)](https://teyou21.medium.com/convert-a-tensorflow-frozen-graph-to-a-tflite-file-part-3-1ccdb3874c4a).
