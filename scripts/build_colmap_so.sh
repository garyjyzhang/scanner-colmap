cd /usr/local/lib/colmap
for staticLib in ./*.a; do
  echo $(basename $staticLib .a).so
  g++ -shared -o $(basename $staticLib .a).so -Wl,--whole-archive $staticLib -Wl,--no-whole-archive
done
