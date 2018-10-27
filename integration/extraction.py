import os.path
import sys

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

db = Database()

cwd = os.path.dirname(os.path.abspath(__file__))
db.load_op(
    os.path.join(cwd, 'op_cpp/build/libextraction_op.so'),
    os.path.join(cwd, 'op_cpp/build/siftExtraction_pb2.py'))

image_ids = db.sources.Column()
frames = db.sources.FrameColumn()

# run SIFT extractions
keypoints, descriptors, cameras = db.ops.SiftExtraction(
    image_ids=image_ids, frames=frames)

output = db.sinks.Column(
    columns={'image_id': image_ids, 'keypoints': keypoints, 'descriptors': descriptors, 'camera': cameras})

job = Job(op_args={
    image_ids: db.table('frames').column('image_id'),
    frames: db.table('frames').column('frame'),
    output: 'extraction'})

output_tables = db.run(output, [job], force=True,
                       io_packet_size=50, work_packet_size=25)
print(db.summarize())
