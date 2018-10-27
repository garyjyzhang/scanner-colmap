import os.path
import sys

from scannerpy import Database, Job
from scannerpy import ProtobufGenerator, Config

db = Database()
cwd = os.path.dirname(os.path.abspath(__file__))

db.load_op(
    os.path.join((cwd), 'op_cpp/build/libprepare_image.so')
)

image_dir = os.path.expanduser(sys.argv[1])
image_paths = []

for i, file in enumerate(os.listdir(image_dir)):
    if os.path.splitext(file)[1] == ".JPG":
        image_paths.append(os.path.join(image_dir, file))

image_paths.sort()

files = db.sources.Files()
frames = db.ops.ImageDecoder(img=files)

# generate image ids
image_ids = db.ops.PrepareImage(frames=frames)

output = db.sinks.Column(
    columns={'image_id': image_ids, 'frame': frames})

job = Job(op_args={
    files: {'paths': image_paths},
    output: 'frames'})

output_tables = db.run(output, [job], force=True)
print(db.summarize())
