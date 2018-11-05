import os.path
import sys

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

arg_parser = argparse.ArgumentParser(
    description='Automatically run the reconstruction on the given images')
arg_parser.add_argument('--scanner_config', dest='scanner_config',
                        help='the path to the scanner config file')
arg_parser.add_argument('--output_table', dest='output_table',
                        help='the name of the output table', default='reconstruction')
arg_parser.add_argument('--image_path', dest='image_path', required=True,
                        help='the path to images, the images should be named appropriately to reflect their orders e.g. image_01.JPG, image_02.JPG')
arg_parser.add_argument(
    '--extraction_packet_size', dest='extraction_packet_size', type=int, default=25, help='the extraction packet size')
arg_parser.add_argument('--matching_overlap', dest='matching_overlap',
                        default=10, help='the matching window size', type=int)
arg_parser.add_argument(
    '--matching_packet_size', dest='matching_packet_size', type=int, default=25, help='the feature matching packet size')
arg_parser.add_argument('--cluster_size', dest='cluster_size',
                        default=10, help='the number of key images to use for each submodel', type=int)
arg_parser.add_argument('--cluster_overlap', dest='cluster_overlap', default=5,
                        type=int, help='the number of key images that adjacent submodels share')
arg_parser.add_argument('--merge_batch_size', dest='merge_batch_size', default=,
                        type=int, help='the number of key images that adjacent submodels share')
args = arg_parser.parse_args()

db = Database(config_path=args.scanner_config)

################################################################################
# Load cpp ops
################################################################################
cwd = os.path.dirname(os.path.abspath(__file__))
db.load_op(
    os.path.join((cwd), 'op_cpp/build/libprepare_image.so'))
db.load_op(
    os.path.join(cwd, 'op_cpp/build/libextraction_op.so'),
    os.path.join(cwd, 'op_cpp/build/siftExtraction_pb2.py'))
db.load_op(
    os.path.join(cwd, 'op_cpp/build/libsequential_matching.so'),
    os.path.join(cwd, 'op_cpp/build/colmap_pb2.py'))
db.load_op(
    os.path.join(cwd, 'op_cpp/build/libincremental_mapping.so'))
db.load_op(
    os.path.join(cwd, 'op_cpp/build/libmerge_mapping.so'))

################################################################################
# prepare images
################################################################################
image_dir = os.path.expanduser(args.image_path)
image_paths = []

for i, file in enumerate(os.listdir(image_dir)):
    if os.path.splitext(file)[1] == ".JPG":
        image_paths.append(os.path.join(image_dir, file))

image_paths.sort()

files = db.sources.Files()
frames = db.ops.ImageDecoder(img=files)

image_ids = db.ops.PrepareImage(frames=frames)

################################################################################
# perform feature extraction
################################################################################

keypoints, descriptors, cameras = db.ops.SiftExtraction(
    image_ids=image_ids, frames=frames)

################################################################################
# perform feature matching
################################################################################
matching_stencil = range(0, args.matching_overlap)
pair_image_ids, two_view_geometries = db.ops.SequentialMatchingCPU(
    image_ids=image_ids, keypoints=keypoints, descriptors=descriptors, stencil=matching_stencil
)

################################################################################
# perform sparse reconstruction
################################################################################
batch_size = args.cluster_size - args.cluster_overlap
matching_stencil = range(0, args.matching_overlap + args.cluster_size)

cluster_ids, cameras, images, points3d = db.ops.IncrementalMappingCPU(
    image_id=image_ids, pair_image_ids=pair_image_ids, two_view_geometries=two_view_geometries, keypoints=keypoints, camera=cameras, batch=batch_size, stencil=matching_stencil)

################################################################################
# remove empty rows from sparse reconstruction
################################################################################


def remove_empty_rows(*input_cols):
    return [db.streams.Stride(col, batch_size) for col in input_cols]


cluster_id, cameras, images, points3d = remove_empty_rows(
    cluster_ids, cameras, images, points3d)

################################################################################
# merge submodels
################################################################################
cluster_id, cameras, images, points3d = db.ops.MergeMappingCPU(
    cluster_id=cluster_id, cameras=cameras, images=images, points3d=points3d, batch=num_submodels)

output = db.sinks.Column(
    columns={'cluster_id': cluster_id, 'cameras': cameras, 'images': images, 'points3d': points3d})

job = Job(op_args={
    files: {'paths': image_paths},
    output: args.output_table})

output_tables = db.run(output, [job], force=True)
print(db.summarize())
