import os.path
import sys

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config
import scannerpy._python as bindings

db = Database()
SEQUENTIAL_MATCHING_OVERLAP = 2

cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join((cwd), 'op_cpp/build/libprepare_image.so')
)

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libextraction_op.so'),
    os.path.join(cwd, 'op_cpp/build/siftExtraction_pb2.py'))

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libsequential_matching.so'),
    os.path.join(cwd, 'op_cpp/build/colmap_pb2.py'))

image_dir = os.path.expanduser(sys.argv[1])
image_paths = []

for i, file in enumerate(os.listdir(image_dir)):
    if os.path.splitext(file)[1] == ".JPG":
        image_paths.append(os.path.join(image_dir, file))

files = db.sources.Files()
frames = db.ops.ImageDecoder(img=files)

# generate image ids
image_ids = db.ops.PrepareImage(frames=frames)
# run SIFT extractions
images, keypoints, descriptors, cameras = db.ops.SiftExtraction(
    image_ids=image_ids, frames=frames)

# run feature matching and two view geometry verification
matching_stencil = range(0, SEQUENTIAL_MATCHING_OVERLAP)

matching_results = db.ops.SequentialMatchingCPU(
    image_ids=image_ids, keypoints=keypoints, descriptors=descriptors, stencil=matching_stencil
)

# write to table
output = db.sinks.Column(
    columns={'image_id': image_ids, 'matching_result': matching_results})

job = Job(op_args={
    files: {'paths': image_paths},
    output: 'matching'})

output_tables = db.run(output, [job], force=True)
print(db.summarize())
