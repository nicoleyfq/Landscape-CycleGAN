mkdir data
mkdir data/sketch
mkdir data/sketch/Harvard
mkdir data/sketch/Metropolitan
mkdir data/sketch/Princeton
mkdir data/sketch/Smithsonian
git clone https://github.com/alicex2020/Chinese-Landscape-Painting-Dataset.git data


mv data/All-Paintings data/paintings

VAR='data/paintings'
unzip $VAR/Harvard/Harvard-1.zip -d $VAR/Harvard/jpg
mv $VAR/Harvard/jpg/Harvard/* $VAR//Harvard/jpg
rm $VAR/Harvard/Harvard-1.zip
rmdir $VAR/Harvard/jpg/Harvard
rm data/dataset-samples.jpg
rm data/paper-figure.jpg
rm data/README.md

unzip $VAR/Metropolitan/Metropolitan-1.zip -d $VAR/Metropolitan/jpg
rm $VAR/Metropolitan/Metropolitan-1.zip
unzip $VAR/Metropolitan/Metropolitan-2.zip -d $VAR/Metropolitan/jpg
rm $VAR/Metropolitan/Metropolitan-2.zip

mv $VAR/Metropolitan/jpg/met-1/* $VAR/Metropolitan/jpg/
mv $VAR/Metropolitan/jpg/met-2/* $VAR/Metropolitan/jpg/
rmdir $VAR/Metropolitan/jpg/met-1
rmdir $VAR/Metropolitan/jpg/met-2


unzip $VAR/Princeton/Princeton-1.zip -d $VAR/Princeton/jpg
rm $VAR/Princeton/Princeton-1.zip
unzip $VAR/Princeton/Princeton-2.zip -d $VAR/Princeton/jpg
rm $VAR/Princeton/Princeton-2.zip

mv $VAR/Princeton/jpg/Princeton-1/* $VAR/Princeton/jpg/
mv $VAR/Princeton/jpg/Princeton-2/* $VAR/Princeton/jpg/
rmdir $VAR/Princeton/jpg/Princeton-1
rmdir $VAR/Princeton/jpg/Princeton-2

find $VAR/Smithsonian -name "*.zip" |  while read filename; do unzip -o -d "`dirname "$filename"`" "$filename"; done;
find $VAR/Smithsonian -name "*.zip" |  while read filename; do rm $filename; done;
mkdir $VAR/Smithsonian/jpg
rm -r $VAR/Smithsonian/__MACOSX

find $VAR/Smithsonian -name "*-*" |  while read filename; do mv $filename/* $VAR/Smithsonian/jpg; done;
find $VAR/Smithsonian -name "*-*" |  while read filename; do rm -r $filename; done;
