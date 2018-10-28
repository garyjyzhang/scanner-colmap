import os.path
import sys
import argparse

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

arg_parser = argparse.ArgumentParser(
    description='Perform SIFT extraction on input images.')
arg_parser.add_argument('--scanner_config', dest='scanner_config',
                        help='the path to the scanner config file')
arg_parser.add_argument('--input_table', dest='input_table', default='frames',
                        help='the input table where the frames and frame ids are stored')
arg_parser.add_argument('--output_table', dest='output_table',
                        help='the name of the output table', default='extraction')
arg_parser.add_argument(
    '--packet_size', dest='packet_size', type=int, default=25, help='the number of frames to dispatch to each extraction kernel')
args = arg_parser.parse_args()

db = Database(config_path=args.scanner_config)

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
    image_ids: db.table(args.input_table).column('image_id'),
    frames: db.table(args.input_table).column('frame'),
    output: args.output_table})

output_tables = db.run(output, [job], force=True,
                       io_packet_size=args.packet_size, work_packet_size=args.packet_size)
print(db.summarize())
