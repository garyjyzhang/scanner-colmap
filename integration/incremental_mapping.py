import os.path
import sys

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

db = Database()

cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libincremental_mapping.so')
)

SEQUENTIAL_MATCHING_OVERLAP = 10
CLUSTER_SIZE = 10
CLUSTER_OVERLAP = 5

batch_size = CLUSTER_SIZE - CLUSTER_OVERLAP

matching_stencil = range(0, SEQUENTIAL_MATCHING_OVERLAP + CLUSTER_SIZE)
num_images = db.table('frames').num_rows()

image_ids = db.sources.Column()
pair_image_ids = db.sources.Column()
two_view_geometries = db.sources.Column()
keypoints = db.sources.Column()
camera = db.sources.Column()

cluster_id, cameras, images, points3d = db.ops.IncrementalMappingCPU(
    image_id=image_ids, pair_image_ids=pair_image_ids, two_view_geometries=two_view_geometries, keypoints=keypoints, camera=camera, batch=batch_size, stencil=matching_stencil)


def remove_empty_rows(*input_cols):
    return [db.streams.Stride(col, CLUSTER_SIZE) for col in input_cols]


# cluster_id, cameras, images, points3d = remove_empty_rows(cluster_id,
#                                                           cameras, images, points3d)

output = db.sinks.Column(
    columns={'cluster_id': cluster_id, 'cameras': cameras, 'images': images, 'points3d': points3d})


job = Job(op_args={
    image_ids: db.table('extraction').column('image_id'),
    pair_image_ids: db.table('matching').column('pair_image_ids'),
    two_view_geometries: db.table('matching').column('two_view_geometries'),
    keypoints: db.table('extraction').column('keypoints'),
    camera: db.table('extraction').column('camera'),
    output: 'mapping'
})

output_tables = db.run(output, [job], force=True)
print(db.summarize())
