# Landscape-CycleGAN
Computer vision project for ECE 285

# Table of Contents  
1. [Folder Organization](#folder)
2. [How to Run Code](#run)  
3. [About Dataset](#data) 


<a name="folder"/></a>
## Folder Organization
1. **Model** : contains all **.py** files for models, **.sh** files for downloading datasets,and running models
2. **Visualization** : contains all **.ipynb** for visualization

<a name="run"/></a>
## How to Run Code
1. Reproduce environment using 
<pre>
conda env create -f environment.yml
</pre>

2. Test all third party modules have been installed 
<pre>
python3 ./library_requirement.py
</pre>

3. Download ./Model folder, insider downloaded unzipped folder, download dataset by running
<pre>
sh get_data.sh
</pre>

4. Reproduce all the sketch or painting images by running following command:
<pre>
sh run_sketch.sh
</pre>
Note: change the sub-dataset name inside shell script to change the training dataset for sketch and painting

<a name="data"/></a>
## About Dataset
The dataset we used for training and testing were from https://github.com/alicex2020/Chinese-Landscape-Painting-Datasetwe, which provided the dataset used to train GAN model. 
The dataset consists of 2,192 high-quality traditional Chinese landscape paintings (中国山水画). All paintings are sized 512x512, from the following sources:
* <a href=https://artmuseum.princeton.edu/search/collections>Princeton University Art Museum</a>, 362 paintings
* <a href=https://harvardartmuseums.org/collections/api>Harvard University Art Museum</a>, 101 paintings
* <a href=https://metmuseum.github.io/>Metropolitan Museum of Art</a>, 428 paintings
* <a href=http://edan.si.edu/openaccess/apidocs/>Smithsonian's Freer Gallery of Art</a>, 1,301 paintings