#! /bin/bash
set -e

# Extract frame triplets from the UCF-101 dataset
# (http://crcv.ucf.edu/data/UCF101.php). Run as follows:
#
#   ./extract-ucf101.sh dir/file.avi
#
# Or, with parallel from moreutils, you can do it for all videos
# over many cores:
#
#   parallel -j 12 ./extract-ucf101.sh -- $( find -name \*.avi )
#
# but do note that this will produce ~250 GB of PNGs, probably many
# more frames than you actually would get to use for training
# and likely straining the file system with ~5M files.
#
# The script will create a set of frame files that you can easily combine
# and use for training:
#
#  for N in 1 2 3; do echo $N; cat $( find -name \*_frame$N.txt | sort -u )  > ../frame$N.txt; done

FILE=$1
PREFIX=$( dirname $FILE )/$( basename $FILE | tr -c "a-zA-Z0-9_-" "_" )

echo $FILE

rm -f ${PREFIX}_frame[123].tmp.txt ${PREFIX}_delayed1_*.png ${PREFIX}_delayed2_*.png

ffmpeg -loglevel error -i $FILE -vf scale=256:256 ${PREFIX}_%04d.png
NUM_FRAMES=$( ls -1 ${PREFIX}_*.png | sed 's/.*\([0-9]\{4\}\)\.png$/\1/' | tail -n 1 | sed 's/^0*//' )

# Verify that the frames contain some sort of motion (we use PSNR <35 dB as a proxy).
# Note that we need to check the entire triplet, since many of the videos seem to have
# been through some sort of pull-up, so there could be differences between n and n+2,
# but n and n+1 would essentially be identical. See v_BoxingPunchingBag_g09_c02.avi
# for an example.
for X in `seq 1 $(( NUM_FRAMES - 2 ))`; do
	ln -s $( basename $( printf "%s_%04d.png" $PREFIX $(( X + 1 )) ) ) $( printf "%s_delayed1_%04d.png" $PREFIX $X )
	ln -s $( basename $( printf "%s_%04d.png" $PREFIX $(( X + 2 )) ) ) $( printf "%s_delayed2_%04d.png" $PREFIX $X )
done	

ffmpeg -loglevel error -i ${PREFIX}_delayed1_%04d.png -i ${PREFIX}_%04d.png -filter_complex "psnr=stats_file=${PREFIX}.d1.psnr" -f null /dev/null
ffmpeg -loglevel error -i ${PREFIX}_delayed1_%04d.png -i ${PREFIX}_delayed2_%04d.png -filter_complex "psnr=stats_file=${PREFIX}.d2.psnr" -f null /dev/null

cut -d" " -f1,6 < ${PREFIX}.d1.psnr > ${PREFIX}.d1.cut.psnr
cut -d" " -f1,6 < ${PREFIX}.d2.psnr > ${PREFIX}.d2.cut.psnr

join ${PREFIX}.d1.cut.psnr ${PREFIX}.d2.cut.psnr | sed 's/psnr_avg://g;s/n://;s/\.[0-9]*//g' | while read FRAMENO PSNR1 PSNR2; do
	#echo FRAMENO=$FRAMENO PSNR1=$PSNR1 PSNR2=$PSNR2
	if [ "$FRAMENO" -le $(( NUM_FRAMES - 2 )) ] && [ "$PSNR1" != "inf" ] && [ "$PSNR1" -lt 35 ] && [ "$PSNR2" != "inf" ] && [ "$PSNR2" -lt 35 ]; then
		echo "generate tmp.txt"
                printf "%s_%04d.png\n" $PREFIX $(( FRAMENO + 0 )) >> ${PREFIX}_frame1.tmp.txt
                printf "%s_%04d.png\n" $PREFIX $(( FRAMENO + 1 )) >> ${PREFIX}_frame2.tmp.txt
                printf "%s_%04d.png\n" $PREFIX $(( FRAMENO + 2 )) >> ${PREFIX}_frame3.tmp.txt
	fi
done

for X in `seq 1 $(( NUM_FRAMES - 2 ))`; do
	rm $( printf "%s_delayed1_%04d.png" $PREFIX $X )
	rm $( printf "%s_delayed2_%04d.png" $PREFIX $X )
done	

mv ${PREFIX}_frame1.tmp.txt ${PREFIX}_frame1.txt
mv ${PREFIX}_frame2.tmp.txt ${PREFIX}_frame2.txt
mv ${PREFIX}_frame3.tmp.txt ${PREFIX}_frame3.txt
rm -f ${PREFIX}.d1.psnr
rm -f ${PREFIX}.d2.psnr
rm -f ${PREFIX}.d1.cut.psnr
rm -f ${PREFIX}.d2.cut.psnr

