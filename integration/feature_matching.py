import os.path
import sys
import argparse

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

arg_parser = argparse.ArgumentParser(
    description='Perform feature matching on the input keypoints.')
arg_parser.add_argument('--scanner_config', dest='scanner_config',
                        help='the path to the scanner config file')
arg_parser.add_argument('--input_table', dest='input_table', default='extraction',
                        help='the input table where the frames and frame ids are stored')
arg_parser.add_argument('--output_table', dest='output_table',
                        help='the name of the output table', default='matching')
arg_parser.add_argument('--overlap', dest='overlap',
                        default=10, help='the matching window size', type=int)
arg_parser.add_argument(
    '--packet_size', dest='packet_size', type=int, default=25, help='the number of frames to dispatch to each feature matching kernel')
args = arg_parser.parse_args()

db = Database(config_path=args.scanner_config)

cwd = os.path.dirname(os.path.abspath(__file__))
db.load_op(
    os.path.join(cwd, 'op_cpp/build/libsequential_matching.so'),
    os.path.join(cwd, 'op_cpp/build/colmap_pb2.py'))

matching_stencil = range(0, args.overlap)

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
    image_ids: db.table(args.input_table).column('image_id'),
    keypoints: db.table(args.input_table).column('keypoints'),
    descriptors: db.table(args.input_table).column('descriptors'),
    camera: db.table(args.input_table).column('camera'),
    output: args.output_table})

output_tables = db.run(output, [job], force=True,
                       io_packet_size=args.packet_size, work_packet_size=args.packet_size)
print(db.summarize())
