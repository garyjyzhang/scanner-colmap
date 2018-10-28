import os.path
import sys
import argparse

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

arg_parser = argparse.ArgumentParser(
    description='Perform SIFT extraction on input images.')
arg_parser.add_argument('--image_path', dest='image_path', required=True,
                        help='the path to images, the images should be named appropriately to reflect their orders e.g. image_01.JPG, image_02.JPG')
arg_parser.add_argument('--output_table', dest='output_table',
                        help='the name of the output table', default='frames')
args = arg_parser.parse_args()

db = Database()
cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join((cwd), 'op_cpp/build/libprepare_image.so')
)

image_dir = os.path.expanduser(args.image_path)
image_paths = []

for i, file in enumerate(os.listdir(image_dir)):
    if os.path.splitext(file)[1] == ".JPG":
        image_paths.append(os.path.join(image_dir, file))

image_paths.sort()

files = db.sources.Files()
frames = db.ops.ImageDecoder(img=files)

# generate image ids
image_ids = db.ops.PrepareImage(frames=frames)

output = db.sinks.Column(
    columns={'image_id': image_ids, 'frame': frames})

job = Job(op_args={
    files: {'paths': image_paths},
    output: args.output_table})

output_tables = db.run(output, [job], force=True)
print(db.summarize())
