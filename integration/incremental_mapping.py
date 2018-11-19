import argparse
import os.path

from scannerpy import Database, Job

arg_parser = argparse.ArgumentParser(
    description=
    ('Perform colmap sparse reconstruction on the input images. the image data a'
     're organized into clusters and used to built submodels that can be merged later on.'
     ))
arg_parser.add_argument(
    '--scanner_config',
    dest='scanner_config',
    help='the path to the scanner config file')
arg_parser.add_argument(
    '--extraction_table',
    dest='extraction_table',
    default='extraction',
    help='the output table of extraction kernel')
arg_parser.add_argument(
    '--matching_table',
    dest='matching_table',
    default='matching',
    help='the output table of feature matching kernel')
arg_parser.add_argument(
    '--output_table',
    dest='output_table',
    help='the name of the output table',
    default='mapping')
arg_parser.add_argument(
    '--matching_overlap',
    dest='matching_overlap',
    default=10,
    help=
    'the matching window size. This should be consistent with the window size used for feature matching',
    type=int)
arg_parser.add_argument(
    '--cluster_size',
    dest='cluster_size',
    default=10,
    help='the number of key images to use for each cluster',
    type=int)
arg_parser.add_argument(
    '--cluster_overlap',
    dest='cluster_overlap',
    default=5,
    type=int,
    help='the number of key images that adjacent cluster share')
args = arg_parser.parse_args()

db = Database(config_path=args.scanner_config)

cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libincremental_mapping.so'),
    os.path.join(cwd, 'op_cpp/build/incremental_mapping_pb2.py'))

step_size = args.cluster_size - args.cluster_overlap

matching_stencil = range(0, args.matching_overlap + args.cluster_size)
num_images = db.table(args.extraction_table).num_rows()

image_ids = db.sources.Column()
pair_image_ids = db.sources.Column()
two_view_geometries = db.sources.Column()
keypoints = db.sources.Column()
camera = db.sources.Column()

cluster_id, cameras, images, points3d = db.ops.IncrementalMappingCPU(
    image_id=image_ids,
    pair_image_ids=pair_image_ids,
    two_view_geometries=two_view_geometries,
    keypoints=keypoints,
    camera=camera,
    stencil=matching_stencil,
    step_size=step_size,
)


def sample(*input_cols):
    return [db.streams.Stride(col, step_size) for col in input_cols]


cluster_id, cameras, images, points3d = sample(cluster_id, cameras, images,
                                               points3d)

output = db.sinks.Column(
    columns={
        'cluster_id': cluster_id,
        'cameras': cameras,
        'images': images,
        'points3d': points3d
    })

job = Job(
    op_args={
        image_ids:
        db.table(args.extraction_table).column('image_id'),
        pair_image_ids:
        db.table(args.matching_table).column('pair_image_ids'),
        two_view_geometries:
        db.table(args.matching_table).column('two_view_geometries'),
        keypoints:
        db.table(args.extraction_table).column('keypoints'),
        camera:
        db.table(args.extraction_table).column('camera'),
        output:
        args.output_table
    })

output_tables = db.run(output, [job], force=True)
print(db.summarize())
