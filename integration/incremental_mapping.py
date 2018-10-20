import os.path
import sys

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

db = Database()

cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libincremental_mapping.so')
)

SEQUENTIAL_MATCHING_OVERLAP = 2
matching_stencil = range(0, SEQUENTIAL_MATCHING_OVERLAP)

image_ids = db.sources.Column()
pair_image_ids = db.sources.Column()
two_view_geometries = db.sources.Column()
keypoints = db.sources.Column()
camera = db.sources.Column()

_ = db.ops.IncrementalMappingCPU(
    image_id=image_ids, pair_image_ids=pair_image_ids, two_view_geometries=two_view_geometries, keypoints=keypoints, camera=camera, stencil=matching_stencil)

output = db.sinks.Column(
    columns={'image_id': _})


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
