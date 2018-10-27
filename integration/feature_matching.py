import os.path
import sys

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

db = Database()

cwd = os.path.dirname(os.path.abspath(__file__))
db.load_op(
    os.path.join(cwd, 'op_cpp/build/libsequential_matching.so'),
    os.path.join(cwd, 'op_cpp/build/colmap_pb2.py'))

SEQUENTIAL_MATCHING_OVERLAP = 10
matching_stencil = range(0, SEQUENTIAL_MATCHING_OVERLAP)

image_ids = db.sources.Column()
keypoints = db.sources.Column()
descriptors = db.sources.Column()
camera = db.sources.Column()

pair_image_ids, two_view_geometries = db.ops.SequentialMatchingCPU(
    image_ids=image_ids, keypoints=keypoints, descriptors=descriptors, stencil=matching_stencil
)

output = db.sinks.Column(
    columns={'pair_image_ids': pair_image_ids, 'two_view_geometries': two_view_geometries})

job = Job(op_args={
    image_ids: db.table('extraction').column('image_id'),
    keypoints: db.table('extraction').column('keypoints'),
    descriptors: db.table('extraction').column('descriptors'),
    camera: db.table('extraction').column('camera'),
    output: 'matching'})

output_tables = db.run(output, [job], force=True,
                       io_packet_size=20, work_packet_size=10)
print(db.summarize())
