CURRENT_DIR=`pwd`
#change output directory 
export OUTPUR_DIR=../data/sketch/landscape

#change subset
python3 sketch.py \
  --subset='landscape' \
  --output_dir=$OUTPUR_DIR
