import os.path
import sys

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

db = Database()

CLUSTER_SIZE = 5

cluster_id_src = db.sources.Column()
cameras_src = db.sources.Column()
images_src = db.sources.Column()
points3d_src = db.sources.Column()


def remove_empty_rows(*input_cols):
    return [db.streams.Stride(col, CLUSTER_SIZE) for col in input_cols]


cluster_id, cameras, images, points3d = remove_empty_rows(
    cluster_id_src, cameras_src, images_src, points3d_src)

output = db.sinks.Column(
    columns={'cluster_id': cluster_id, 'cameras': cameras, 'images': images, 'points3d': points3d})

job = Job(op_args={
    cluster_id_src: db.table('mapping').column('cluster_id'),
    cameras_src: db.table('mapping').column('cameras'),
    images_src: db.table('mapping').column('images'),
    points3d_src: db.table('mapping').column('points3d'),
    output: 'submodel'
})

output_tables = db.run(output, [job], force=True)
print(db.summarize())
