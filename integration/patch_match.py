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
    '--input_table',
    dest='input_table',
    help='the name of the input table',
    default='prepare_patch_match')
arg_parser.add_argument(
    '--num_overlap',
    dest='overlap',
    type=int,
    default=10,
    help="the number of overlapping images to use for each reference image")
arg_parser.add_argument(
    '--output_table',
    dest='output_table',
    help='the name of the output table',
    default='patch_match')
args = arg_parser.parse_args()

db = Database(config_path=args.scanner_config)

cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libpatch_match.so'))

num_images = db.table(args.input_table).num_rows()

R = db.sources.Column()
T = db.sources.Column()
K = db.sources.Column()
width = db.sources.Column()
height = db.sources.Column()
scan_width = db.sources.Column()
bitmap = db.sources.Column()
depth_min = db.sources.Column()
depth_max = db.sources.Column()
image_id = db.sources.Column()

def get_partition_ranges(num, step):
    ranges = []
    r = list(range(0, num_images, step))
    for i in range(1, len(r)):
        ranges.append((r[i - 1], r[i]))

    if ranges[-1][1] != num_images - 1:
        ranges.append((r[-1], num_images - 1))

    return ranges

# slice input
partitioner = db.partitioner.ranges(get_partition_ranges(num_images, 10))
# R_ = db.streams.Slice(R, partitioner=partitioner)
# T_ = db.streams.Slice(T, partitioner=partitioner)
# K_ = db.streams.Slice(K, partitioner=partitioner)
# width_ = db.streams.Slice(width, partitioner=partitioner)
# height_ = db.streams.Slice(height, partitioner=partitioner)
# scan_width_ = db.streams.Slice(scan_width, partitioner=partitioner)
# bitmap_ = db.streams.Slice(bitmap, partitioner=partitioner)
# depth_min_ = db.streams.Slice(depth_min, partitioner=partitioner)
# depth_max_ = db.streams.Slice(depth_max, partitioner=partitioner)
# image_id_ = db.streams.Slice(image_id, partitioner=partitioner)

R_ = db.streams.Range(R, start=0, end=10)
T_ = db.streams.Range(T, start=0, end=10)
K_ = db.streams.Range(K, start=0, end=10)
width_ = db.streams.Range(width, start=0, end=10)
height_ = db.streams.Range(height, start=0, end=10)
scan_width_ = db.streams.Range(scan_width, start=0, end=10)
bitmap_ = db.streams.Range(bitmap, start=0, end=10)
depth_min_ = db.streams.Range(depth_min, start=0, end=10)
depth_max_ = db.streams.Range(depth_max, start=0, end=10)
image_id_ = db.streams.Range(image_id, start=0, end=10)


depth_map, normal_map = db.ops.PatchMatch(
    R=R_, T=T_, K=K_, width=width_, height=height_, scan_width=scan_width_,
    bitmap=bitmap_, depth_min=depth_min_, depth_max=depth_max_, image_id=image_id_, stencil=range(0, args.overlap))

# depth_map, normal_map = db.streams.Unslice(depth_map), db.streams.Unslice(normal_map)

output = db.sinks.Column(
    columns={
        'depth_map': depth_map,
        'normal_map': normal_map,
    })

job = Job(
    op_args={
        R: db.table(args.input_table).column('R'),
        T: db.table(args.input_table).column('T'),
        K: db.table(args.input_table).column('K'),
        width: db.table(args.input_table).column('width'),
        height: db.table(args.input_table).column('height'),
        scan_width: db.table(args.input_table).column('scan_width'),
        bitmap: db.table(args.input_table).column('bitmap'),
        depth_min: db.table(args.input_table).column('depth_min'),
        depth_max: db.table(args.input_table).column('depth_max'),
        image_id: db.table(args.input_table).column('image_id'),
        output: args.output_table,
    })

output_tables = db.run(output, [job], force=True)
print(db.summarize())
