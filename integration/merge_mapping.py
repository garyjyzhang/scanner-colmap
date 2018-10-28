import os.path
import sys
import math
import argparse

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

arg_parser = argparse.ArgumentParser(
    description='Merge all the submodels from the input table, currently performs linear merging on all the submodels')
arg_parser.add_argument('--scanner_config', dest='scanner_config',
                        help='the path to the scanner config file')
arg_parser.add_argument('--input_table', dest='input_table', default='submodels',
                        help='the input table where submodels are stored')
arg_parser.add_argument('--output_table', dest='output_table',
                        help='the name of the output table', default='models')
args = arg_parser.parse_args()

db = Database(config_path=args.scanner_config)

cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libmerge_mapping.so')
)

num_submodels = db.table(args.input_table).num_rows()
print("num submodels: %d" % num_submodels)

cluster_id_src = cluster_id = db.sources.Column()
cameras_src = cameras = db.sources.Column()
images_src = images = db.sources.Column()
points3d_src = points3d = db.sources.Column()

cluster_id, cameras, images, points3d = db.ops.MergeMappingCPU(
    cluster_id=cluster_id, cameras=cameras, images=images, points3d=points3d, batch=num_submodels)

output = db.sinks.Column(
    columns={'cluster_id': cluster_id, 'cameras': cameras, 'images': images, 'points3d': points3d})

job = Job(op_args={
    cluster_id_src: db.table(args.input_table).column('cluster_id'),
    cameras_src: db.table(args.input_table).column('cameras'),
    images_src: db.table(args.input_table).column('images'),
    points3d_src: db.table(args.input_table).column('points3d'),
    output: args.output_table
})

output_tables = db.run(output, [job], force=True)
print(db.summarize())
