colmap_extraction() {
  colmap feature_extractor --database_path colmap.db --image_path $1 --SiftExtraction.use_gpu 0
}

sh /app/scripts/build_colmap_so.sh
cd /app/integration/op_cpp/build
make
cd /app
/bin/bash
