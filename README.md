# scanner-colmap

Scanner-colmap re-expresses Colmap's 3D reconstruction pipeline using the Scanner video/image analysis system. The combination allows Colmap to scale to large image collections or even directly from video inputs. The tool is still under development and improvements, the current progress is up to the sparse reconstruction step. At this moment, scanner-colmap can only be run with CPU's. 

![Alt text](https://user-images.githubusercontent.com/12142904/47631345-f57fa180-db02-11e8-833e-a1134f51fb9b.png)
The resulting model of running scanner-colmap on the [Gerrard Hall image set](https://drive.google.com/drive/folders/0B6q7-Pen0AbDTk5WM2hkUjF0Znc)

## Installation:
The tool can be accessed from the pre-built docker image:
```
docker pull garyzhang830/scanner-colmap
```
And then inside the container:
```
cd /opt/scanner-colmap/integration
python3 prepare_image.py --image_path /path/to/image/ --scanner_config /path/to/config.toml
```
It is recommended that you mount a volume containing your project files such as image data and the scanner config file in your docker container so that they are persisted across different docker runs.


Alternatively, you can choose to install the dependencies and build scanner-colmap from source. First make sure you have Scanner and Colmap installed:
- [Scanner](http://scanner.run/installation.html)
- [Colmap](https://colmap.github.io/install.html)

Then, clone this repository and do:
```
cd integration/op_cpp
mkdir build 
cd build
cmake ..
make
```

## Usage:
All stages of the 3D reconstruction can be initiated with the task python scripts located at `integration/`. Follow these steps to perform 3D reconstruction on your image set:
1. Image preparation. This step reads in all the images in the given path and tag them with a unque id.
```
python3 prepare_image.py --image_path /path/to/images --scanner_config /path/to/config.toml
```

2. Feature extraction. In this step, features of each image are extracted using SIFT and stored into the output table. To speed up the process, set a small _packet_size_ parameter to enable smaller unit of work in Scanner.
```
python3 extraction.py --scanner_config /path/to/config.toml --packet_size 5
```
3. Feature matching. The image features from the previous steps are used to find similarities between pairs of images and generate two view geometries. Each image is matched with the next _overlap_ images in order. A small _packet_size_ is recommended to parallelize the process.
```
python3 feature_matching.py --scanner_config /path/to/config.toml --overlap 10 --packet_size 4
```
4. Sparse reconstruction. In this step, the geometries from the previous step are merged to create sparse 3D models. The number of submodels can be controlled using the _cluster_size_ and _cluster_overlap_ parameters. The _cluster_size_ is the number of key images to use per cluster. The two view geometries of these key images obtained from last step will be unpacked and used to reconstruct the submodel. The _cluster_overlap_ specifies how many key images are shared between each submodel, this is can be increased if model merging fails in the next step.
```
python3 incremental_mapping.py --scanner_config /path/to/config.toml --matching_overlap 10 --cluster_size 10 --cluster_overlap 5
```

## Image data
If you are looking for some data to start with, Colmap provides some pretty cool [image sets](https://colmap.github.io/datasets.html)
