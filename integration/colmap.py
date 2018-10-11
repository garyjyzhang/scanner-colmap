from scannerpy import Database, Job
import os.path
from scannerpy import ProtobufGenerator, Config
import scannerpy._python as bindings

# cfg = Config()
# protobufs = ProtobufGenerator(cfg)
# mp = protobufs.MachineParameters()
# mp2 = bindings.default_machine_params()
# mp.ParseFromString(mp2)
# mp.num_cpus = 5
# mp.num_load_workers = 5
# mp.num_save_workers = 5
# print(mp)

# db = Database(machine_params=mp.SerializeToString())
NUM_JOBS = 1

db = Database()

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(cwd, 'op_cpp/build/libextraction_op.so')):
    print(
        'You need to build the custom op first: \n')
    exit()

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libextraction_op.so'),
    os.path.join(cwd, 'op_cpp/build/siftExtraction_pb2.py'))

db.load_op(
    os.path.join(cwd, 'op_cpp/build/libsequential_matching.so'),
    os.path.join(cwd, 'op_cpp/build/colmap_pb2.py'))

image_dir = os.path.abspath(os.path.join(cwd, '../gerrard-hall/images'))
image_paths = []
output_paths = []

for i, file in enumerate(os.listdir(image_dir)):
    if os.path.splitext(file)[1] == ".JPG":
        image_paths.append(os.path.join(image_dir, file))

images = db.sources.Files()
frame = db.ops.ImageDecoder(img=images)

image_ids, keypoints, descriptors = db.ops.SiftExtraction(frame=frame)

# encoded_frame = db.ops.ImageEncoder(frame=extraction, format='jpg')
# output = db.sinks.Files(input=encoded_frame)

pair_id, feature_matches, two_view_geometry = db.ops.SequentialMatchingCPU(
    image_ids=image_ids, keypoints=keypoints, descriptors=descriptors, stencil=[
        0, 1]
)

output = db.sinks.Column(
    columns={'pair_id': pair_id, 'feature_matches': feature_matches, 'descriptors': two_view_geometry})


# SEQUENTIAL_MATCHING_OVERLAP = 10
# matching_results = []
# for i in range(0, SEQUENTIAL_MATCHING_OVERLAP):
#     matching_result.append(db.ops.FeatureMatching())

# output = db.sinks.Column(
#     columns={'keypoints': match_frames})
job = Job(op_args={
    images: {'paths': image_paths},
    # output: {'paths': output_paths},
    output: 'matching'})

output_tables = db.run(output, [job], force=True)
print(db.summarize())
# output_table.profiler().write_trace('keypoints.trace')
