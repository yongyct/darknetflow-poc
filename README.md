# Darknetflow POC (WIP)
POC Implementation with References to Darknet (Darkflow)

# Getting Started
## Installation
After cloning this git repository into your machine, at the root directory, run the following to install the repository's packages to your python environment:
* `pip install -e .`

## Simple Usage
### Configurations
Under `conf` folder, edit the properties as per your machine's settings:
* `ops`: one of the values in ["predict", "train"]
* `in_data_dir`: input data directory
* `out_data_dir`: output data directory
* `weights_dir`: folder to save/load tensorflow checkpoints
* `labels_path`: path to class labels file
* `use_gpu`: boolean value of whether gpu should be used
* `batch_size`: batch size
* `n_epochs`: number of epochs, each iterating through the whole dataset
* `save_interval`: number of steps before saving weights checkpoint
* `keras_weights_path`: path to keras weights, if using keras weights
* `object_threshold`: threshold value above which object will be considered
* `nms_threshold`: threshold value used when selecting nms boxes
* `height`: height in pixels that input should be resized to before feeding to model
* `width`: width in pixels that input should be resized to before feeding to model
* `channels`: channels of input data, usually 3
* `n_classes`: number of classes in the file defined under `labels_path`

# TODO
## Ops Modes
* predict (Keras) - Include option for webcam, videos
* predict (Tensorflow) - Only compatible with TF2.0 checkpoints
* train (Tensorflow) - Only compatible with TF2.0 checkpoints

# References (Many thanks to the below)
* https://github.com/xiaochus/YOLOv3
* https://pjreddie.com/darknet/yolo/
* https://github.com/pjreddie/darknet
* https://github.com/thtrieu/darkflow

# Citation
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

# Built-With
* [OpenCV](https://opencv.org/)
* [Keras](https://keras.io/)
* [Tensorflow 2.0](https://www.tensorflow.org/)
