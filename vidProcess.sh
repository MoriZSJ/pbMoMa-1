## vidProcess
for name in ./test/*.avi; do
  ffmpeg -r 10 -i $name -vf scale=1064:-1 ./test/test.avi
done