# scanner-colmap

Scanner-colmap implements Colmap's 3D reconstruction pipeline into the Scanner video/image analysis system. The combination allows Colmap to scale to large image collections or even directly from video inputs. The tool is still under development, the current progress is up to the sparse reconstruction step. At this moment, scanner-colmap can only be run with CPU's. 

**Installation:**
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
[Scanner](http://scanner.run/installation.html)
[Colmap](https://colmap.github.io/install.html)
