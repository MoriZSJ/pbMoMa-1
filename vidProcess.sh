## vidProcess
for name in ./test/*.avi; do
  ffmpeg -i $name -r 25 -vf scale=1064:-1 ./test/1.avi
done