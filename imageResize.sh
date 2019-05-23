## img resize (3 channel)
for name in cache/*.png; do
  convert -resize 320x320! $name $name
done

## change file type
#mogrify -format png *.gif