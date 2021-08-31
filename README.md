# Jay Patel 2020 Mask R-CNN

This is my implementation of MaskRCNN for cell detection.

The dataset I used is is in [main/dataset](main/dataset). The annotations for each folder (train/val) is in that respective folder in a file named via_export_json.json ([train](main/dataset/train/via_export_json.json) [val](main/dataset/val/via_export_json.json)).

To annotate the dataset I used [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/).

I set up the cell detection class to read from two classes, trap, and cell. Using VGG, I added an attribute with the name **name**. It's type is **checkbox**, then I added the options, *cell*. I also selected toggle the annotation editor to select each annotation per object. Make sure when you're finished you export it to json. The filename must match via_export_json.json and must be in the proper folder (train or val).


You can visualize this dataset masks in with the [Visualize Jupyter Lab](main/Visualize.ipynb).
You can train this dataset with the [Train Jupyter Lab](main/Train.ipynb).
You can predict your own model using the [Predict Jupyter Lab](main/Predict.ipynb).


I was not able to upload my trained models because it is a large file. You will have to modify the model location and name on the predict jupyter lab to match your model.

Finally, all of the settings for running the model including how I loaded the annotations are in [Cell.py](main/Cell.py). This is where you will change the GPU usage, as well as the anchor points and minimum confidence levels.
