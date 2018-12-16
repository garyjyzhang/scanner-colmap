import argparse
import math
import os.path
import sys

from scannerpy import Config, Database, Job, ProtobufGenerator, DeviceType

arg_parser = argparse.ArgumentParser(
    description=
    'Read sparse reconstruction and prepare context for patch match process.'
)
arg_parser.add_argument(
    '--scanner_config',
    dest='scanner_config',
    help='the path to the scanner config file')
arg_parser.add_argument(
    '--input_path',
    dest='input_path',
    help="the path to reconstruction folder, must contain a 'sparse' subfolder containing the sparse reconstruction result")
arg_parser.add_argument(
    '--num_reg_images',
    dest='num_reg_images',
    type=int,
    help="number of registered images in reconstruction")
arg_parser.add_argument(
    '--output_table',
    dest='output_table',
    help='the name of the output table',
    default='prepare_patch_match')
args = arg_parser.parse_args()

db = Database(config_path=args.scanner_config)

cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libprepare_patch_match.so'),
    os.path.join(cwd, 'op_cpp/build/prepare_patch_match_pb2.py'))

image_id = db.sources.Column()

# image_id_repeated = db.streams.RepeatNull(image_id, args.num_reg_images)
image_id_sampled = db.streams.Range(input=image_id, start=0, end=args.num_reg_images)
R, T, K, width, height, scan_width, bitmap, depth_min, depth_max, id = db.ops.PreparePatchMatch(
    sparse_reconstruction_path=args.input_path, batch=args.num_reg_images, image_id=image_id_sampled)

output = db.sinks.Column(
    columns={
        'R': R,
        'T': T,
        'K': K,
        'width': width,
        'height': height,
        'scan_width': scan_width,
        'bitmap': bitmap,
        'depth_min': depth_min,
        'depth_max': depth_max,
        'image_id': id,
    })

job = Job(
    op_args={
        image_id: db.table('frames').column('image_id'),
        output: args.output_table,
    })

output_tables = db.run(output, [job], force=True, work_packet_size=args.num_reg_images, io_packet_size=args.num_reg_images)
print(db.summarize())
