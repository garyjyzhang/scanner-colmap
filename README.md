# scanner-colmap

Scanner-colmap re-expresses [Colmap](https://colmap.github.io/index.html)'s 3D reconstruction pipeline using the [Scanner](https://github.com/scanner-research/scanner) video/image analysis system. The combination allows Colmap to scale to large image collections or even directly from video inputs. The tool is still under development and undergoing improvements. The current progress is up to the stereo fusion step. At this moment, processes before dense reconstruction can only be run with CPU's, the corresponding GPU kernels for these processes will be updated in the future. 

![Alt text](https://user-images.githubusercontent.com/12142904/47631345-f57fa180-db02-11e8-833e-a1134f51fb9b.png)
The resulting model of running sparse reconstruction on the [Gerrard Hall image set](https://drive.google.com/drive/folders/0B6q7-Pen0AbDTk5WM2hkUjF0Znc) using scanner-colmap.

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
3. Feature matching. The image features from the previous steps are used to find similarities between pairs of images and generate two view geometries. Each image is matched with the next _overlap_ number of images in order. A small _packet_size_ is recommended to parallelize the process, given there is sufficient memory. 
```
python3 feature_matching.py --scanner_config /path/to/config.toml --overlap 10 --packet_size 4
```
4. Sparse reconstruction. In this step, the geometries from the previous step are merged to create sparse 3D models. The number of submodels can be controlled using the _cluster_size_ and _cluster_overlap_ parameters. The _cluster_size_ is the number of reference images to use per cluster, where each reference image is a row from the feature matching step. The two view geometries between the key images and their peer images obtained from last step will be unpacked and used to reconstruct the submodel. The _cluster_overlap_ specifies how many key images are shared between each submodel, this is can be increased if model merging fails in the next step.
```
python3 incremental_mapping.py --scanner_config /path/to/config.toml --matching_overlap 10 --cluster_size 10 --cluster_overlap 5
```
Note: The following steps require a gpu and CUDA to run

5. Prepare for dense reconstruction. Before we can perform dense reconstruction using MVS, we need to prepare the necessary input. In this part of the pipeline, we read the sparse reconstruction result and extract image information along with the matrices necessary for dense reconstruction. 
```
python3 prepare_patch_match.py --scanner_config /path/to/config.toml --input_path /path/to/sparse_reconstruction --num_reg_images <number of registered images from sparse reconstruction>
```
Due to the fact that scanner needs to know the output size in advance, we need to pass in the number of registered images from sparse reconstrcuction in order to produce the correct number of rows. 

6. Dense reconstruction. Run dense reconstruction on the prepared inputs from previous step. This step produces a depth and normal map for each registered image based on reference points from the sparse reconstruction.
```
python3 patch_match.py --scanner_config /path/to/config.toml --overlap 20
```
7. Stereo fusion. Produce a colored dense point cloud by fusing and merging the depth maps obtained in the previous step. Each rows in the output is the collection of points for one image, where each point corresponds to one pixel. 
```
python3 stereo_fusion.py --scanner_config /path/to/config.toml --overlap 20
```
8. Poisson meshing. Coming soon... 
## Image data
If you are looking for some data to start with, Colmap provides some pretty cool [image sets](https://colmap.github.io/datasets.html)
