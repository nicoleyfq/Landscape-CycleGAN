CURRENT_DIR=`pwd`
export OUTPUR_DIR=$CURRENT_DIR/data/sketch

python3 sketch.py \
  --subset='Princeton' \
  --output_dir=$OUTPUR_DIR
