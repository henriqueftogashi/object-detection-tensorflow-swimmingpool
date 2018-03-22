# object-detection-tensorflow-swimmingpool

This project is a tutorial to get started with the Tensorflow Object Detection API. Swimming pools are a tricky object to identify and results with limited training (less than 30 min) were not very accurate. I would recommend to train your model for much longer with a much larger training dataset. Depending on the object of choice, good results can be obtained with not much training, but this was not the case for swimming pools.  

## Getting Started

Installation instructions at:

```
https://github.com/tensorflow/models.git 
```

### Folder structure

```
+swimmingpool_model
	-annotations
		labels xmls
	-data
		record files 
		labels csv
	-images
		raw images
	-object_detection
		copied from API model folder
	-slim
		copied from API model folder
	-ssd_mobilenet_v1_coco_2017_11_17
		downloaded from internet
	-test_images
		test_images
	-trained_model
		target folder for the new trained model
	-training
		-model.ckpt (3 files)
		-ssd_mobilenet_v1_pets.config (edit this file - check original online)
		-object-detection.pbtxt (edit this file - check original online)
__init__ (not sure if it is necessary)
generate_tfrecord.py
train.py
trainer.py
xml_to_csv.py
object_detection_tutorial_edited.ipynb
export_inference_graph
```

### Label images (create annotations in xml) with labelImg

```
https://github.com/tzutalin/labelImg
```


### Create data tfrecords (.record)
Use xml_to_csv.py to generate labels_csv then generate_tfrecord.py 

### For tensorflow 1.5 and above:
edit 'object_detection\data_decoders\tf_example_decoder.py' line 110 to remove dct_method=dct_method

### Set PATH Terminal (Windows version):

```
$set PYTHONPATH=%PYTHONPATH%;C:\Users\hfurstenautogashi\Desktop\swimmingpool_model\slim
```

## Training
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

### Tips
If you are running out of memory and this is causing training to fail, try adding the arguments

```
batch_queue_capacity: 2
prefetch_queue_capacity: 2
```

to your config file in the train_config section. For example, placing the two lines between gradient_clipping_by_norm and fine_tune_checkpoint will work. The number 2 above should only be starting values to get training to begin. The default for those values are 8 and 10 respectively and increasing those values should help speed up training.

## Running the trained model
select check point model.ckpt

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix training/model.ckpt-1230 --output_directory trained_model
```

run object_detection_tutorial_edited.ipynb


## good resources / Acknowledgements

These guys helped me a lot:

https://pythonprogramming.net/training-custom-objects-tensorflow-object-detection-api-tutorial/
https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9


