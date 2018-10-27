import os.path
import sys
import math

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

db = Database()

cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libmerge_mapping.so')
)

batch_size = 20
num_submodels = db.table('submodel').num_rows()
print("num submodels: %d" % num_submodels)

cluster_id_src = cluster_id = db.sources.Column()
cameras_src = cameras = db.sources.Column()
images_src = images = db.sources.Column()
points3d_src = points3d = db.sources.Column()


def remove_empty_rows(*input_cols):
    return [db.streams.Stride(col, batch_size) for col in input_cols]


cluster_id, cameras, images, points3d = db.ops.MergeMappingCPU(
    cluster_id=cluster_id, cameras=cameras, images=images, points3d=points3d, batch=batch_size)

# while num_submodels > 1:
#     cluster_id, cameras, images, points3d = db.ops.MergeMappingCPU(
#         cluster_id=cluster_id, cameras=cameras, images=images, points3d=points3d, batch=batch_size)
#
#     cluster_id, cameras, images, points3d = remove_empty_rows(
#         cluster_id, cameras, images, points3d)
#
#     num_submodels = math.ceil(num_submodels / batch_size)

output = db.sinks.Column(
    columns={'cluster_id': cluster_id, 'cameras': cameras, 'images': images, 'points3d': points3d})


job = Job(op_args={
    cluster_id_src: db.table('submodel').column('cluster_id'),
    cameras_src: db.table('submodel').column('cameras'),
    images_src: db.table('submodel').column('images'),
    points3d_src: db.table('submodel').column('points3d'),
    output: 'model'
})

output_tables = db.run(output, [job], force=True)
print(db.summarize())
