## vidProcess
for name in ./test/*.avi; do
  ffmpeg -i $name -r 25 -vf scale=1064:-1 ./test/1.avi
done


#cut video 
# ffmpeg -ss 00:0:00 -t 00:00:02 -i eye-ud.mp4 test.avi
# ffmpeg -ss 00:00:00 -t 00:00:30 -i test.mp4 -vcodec copy -acodec copy output.mp4