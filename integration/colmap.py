import os.path
import sys

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

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

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libincremental_mapping.so')
)
matching_stencil = range(0, SEQUENTIAL_MATCHING_OVERLAP)

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
keypoints, descriptors, cameras = db.ops.SiftExtraction(
    image_ids=image_ids, frames=frames)

# run feature matching and two view geometry verification

pair_image_ids, two_view_geometries = db.ops.SequentialMatchingCPU(
    image_ids=image_ids, keypoints=keypoints, descriptors=descriptors, stencil=matching_stencil
)

# write to table
output = db.sinks.Column(
    columns={'image_id': image_ids, 'pair_image_ids': pair_image_ids, 'two_view_geometries': two_view_geometries, 'keypoints': keypoints, 'camera': cameras})

# image_ids = db.sources.Column()
# pair_image_ids = db.sources.Column()
# two_view_geometries = db.sources.Column()
# keypoints = db.sources.Column()
# camera = db.sources.Column()
#
# _ = db.ops.IncrementalMappingCPU(
#     image_id=image_ids, pair_image_ids=pair_image_ids, two_view_geometries=two_view_geometries, keypoints=keypoints, camera=camera, stencil=matching_stencil)

# output = db.sinks.Column(
#     columns={'image_id': _})

job = Job(op_args={
    files: {'paths': image_paths},
    output: 'matching'})

# job = Job(op_args={
#     image_ids: db.table('matching').column('image_id'),
#     pair_image_ids: db.table('matching').column('pair_image_ids'),
#     two_view_geometries: db.table('matching').column('two_view_geometries'),
#     keypoints: db.table('matching').column('keypoints'),
#     camera: db.table('matching').column('camera'),
#     output: 'mapping'
# })

output_tables = db.run(output, [job], force=True)
print(db.summarize())
