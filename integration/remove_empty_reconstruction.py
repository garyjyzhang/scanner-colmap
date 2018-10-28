import os.path
import sys
import argparse

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

arg_parser = argparse.ArgumentParser(
    description='Perform feature matching on the input keypoints.')
arg_parser.add_argument('--scanner_config', dest='scanner_config',
                        help='the path to the scanner config file')
arg_parser.add_argument('--input_table', dest='input_table', default='mapping',
                        help='the input table')
arg_parser.add_argument('--output_table', dest='output_table',
                        help='the name of the output table', default='submodels')
arg_parser.add_argument('--stride', dest='stride', required=True,
                        help='the sampling frequency', type=int)
args = arg_parser.parse_args()

db = Database(config_path=args.scanner_config)

cluster_id_src = db.sources.Column()
cameras_src = db.sources.Column()
images_src = db.sources.Column()
points3d_src = db.sources.Column()


def remove_empty_rows(*input_cols):
    return [db.streams.Stride(col, args.stride) for col in input_cols]


cluster_id, cameras, images, points3d = remove_empty_rows(
    cluster_id_src, cameras_src, images_src, points3d_src)

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
