{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'tensorflow'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-1-fb87d22d65ce>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     20\u001b[0m \u001b[0;31m# Import Mask RCNN\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     21\u001b[0m \u001b[0msys\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mROOT_DIR\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# To find local version of the library\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 22\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mmrcnn\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mutils\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     23\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mmrcnn\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mvisualize\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     24\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mmrcnn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvisualize\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mdisplay_images\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/github/Peyton2020MaskRCNN/main/Mask_RCNN-master 3/mrcnn/utils.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mrandom\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 16\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mtensorflow\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     17\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mscipy\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     18\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mskimage\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolor\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'tensorflow'"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import sys\n",
    "import itertools\n",
    "import math\n",
    "import logging\n",
    "import json\n",
    "import re\n",
    "import random\n",
    "from collections import OrderedDict\n",
    "import numpy as np\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.patches as patches\n",
    "import matplotlib.lines as lines\n",
    "from matplotlib.patches import Polygon\n",
    "\n",
    "# Root directory of the project\n",
    "ROOT_DIR = 'Mask_RCNN-master 3'\n",
    "\n",
    "# Import Mask RCNN\n",
    "sys.path.append(ROOT_DIR)  # To find local version of the library\n",
    "from mrcnn import utils\n",
    "from mrcnn import visualize\n",
    "from mrcnn.visualize import display_images\n",
    "import mrcnn.model as modellib\n",
    "from mrcnn.model import log\n",
    "\n",
    "import Cell"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "model_dir = \"../logs/\"\n",
    "model_file = \"coco.h5\"\n",
    "coco_path = os.path.abspath(model_dir + model_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Logging before flag parsing goes to stderr.\n",
      "W0420 01:49:55.416901 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.\n",
      "\n",
      "W0420 01:49:55.426159 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.\n",
      "\n",
      "W0420 01:49:55.475694 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.\n",
      "\n",
      "W0420 01:49:55.524517 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:1919: The name tf.nn.fused_batch_norm is deprecated. Please use tf.compat.v1.nn.fused_batch_norm instead.\n",
      "\n",
      "W0420 01:49:55.529148 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:3976: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.\n",
      "\n",
      "W0420 01:49:56.999320 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:2018: The name tf.image.resize_nearest_neighbor is deprecated. Please use tf.compat.v1.image.resize_nearest_neighbor instead.\n",
      "\n",
      "W0420 01:49:57.587314 4521070016 deprecation.py:323] From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/ops/array_ops.py:1475: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.where in 2.0, which has the same broadcast rule as np.where\n",
      "W0420 01:49:57.760549 4521070016 module_wrapper.py:139] From Mask_RCNN-master 3/mrcnn/model.py:553: The name tf.random_shuffle is deprecated. Please use tf.random.shuffle instead.\n",
      "\n",
      "W0420 01:49:57.838016 4521070016 module_wrapper.py:139] From Mask_RCNN-master 3/mrcnn/utils.py:202: The name tf.log is deprecated. Please use tf.math.log instead.\n",
      "\n",
      "W0420 01:49:57.866270 4521070016 deprecation.py:506] From Mask_RCNN-master 3/mrcnn/model.py:600: calling crop_and_resize_v1 (from tensorflow.python.ops.image_ops_impl) with box_ind is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "box_ind is deprecated, use box_indices instead\n"
     ]
    }
   ],
   "source": [
    "model = modellib.MaskRCNN(mode=\"training\", config=Cell.config, model_dir=model_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "if not os.path.exists(coco_path):\n",
    "    utils.download_trained_weights(coco_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0420 01:50:00.497350 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:174: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.\n",
      "\n",
      "W0420 01:50:00.498685 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:181: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.\n",
      "\n",
      "W0420 01:50:00.499603 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:186: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.\n",
      "\n",
      "W0420 01:50:00.527108 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:190: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.\n",
      "\n",
      "W0420 01:50:00.528795 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:199: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.\n",
      "\n",
      "W0420 01:50:02.054198 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:206: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "model.load_weights(coco_path, by_name=True, exclude=[\n",
    "            \"mrcnn_class_logits\", \"mrcnn_bbox_fc\",\n",
    "            \"mrcnn_bbox\", \"mrcnn_mask\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training network heads\n",
      "\n",
      "Starting at epoch 0. LR=0.001\n",
      "\n",
      "Checkpoint Path: ../logs/cell20200420T0150/mask_rcnn_cell_{epoch:04d}.h5\n",
      "Selecting layers to train\n",
      "fpn_c5p5               (Conv2D)\n",
      "fpn_c4p4               (Conv2D)\n",
      "fpn_c3p3               (Conv2D)\n",
      "fpn_c2p2               (Conv2D)\n",
      "fpn_p5                 (Conv2D)\n",
      "fpn_p2                 (Conv2D)\n",
      "fpn_p3                 (Conv2D)\n",
      "fpn_p4                 (Conv2D)\n",
      "In model:  rpn_model\n",
      "    rpn_conv_shared        (Conv2D)\n",
      "    rpn_class_raw          (Conv2D)\n",
      "    rpn_bbox_pred          (Conv2D)\n",
      "mrcnn_mask_conv1       (TimeDistributed)\n",
      "mrcnn_mask_bn1         (TimeDistributed)\n",
      "mrcnn_mask_conv2       (TimeDistributed)\n",
      "mrcnn_mask_bn2         (TimeDistributed)\n",
      "mrcnn_class_conv1      (TimeDistributed)\n",
      "mrcnn_class_bn1        (TimeDistributed)\n",
      "mrcnn_mask_conv3       (TimeDistributed)\n",
      "mrcnn_mask_bn3         (TimeDistributed)\n",
      "mrcnn_class_conv2      (TimeDistributed)\n",
      "mrcnn_class_bn2        (TimeDistributed)\n",
      "mrcnn_mask_conv4       (TimeDistributed)\n",
      "mrcnn_mask_bn4         (TimeDistributed)\n",
      "mrcnn_bbox_fc          (TimeDistributed)\n",
      "mrcnn_mask_deconv      (TimeDistributed)\n",
      "mrcnn_class_logits     (TimeDistributed)\n",
      "mrcnn_mask             (TimeDistributed)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0420 01:50:05.580487 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.\n",
      "\n",
      "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.\n",
      "  \"Converting sparse IndexedSlices to a dense Tensor of unknown shape. \"\n",
      "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.\n",
      "  \"Converting sparse IndexedSlices to a dense Tensor of unknown shape. \"\n",
      "/usr/local/lib/python3.7/site-packages/tensorflow_core/python/framework/indexed_slices.py:424: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.\n",
      "  \"Converting sparse IndexedSlices to a dense Tensor of unknown shape. \"\n",
      "W0420 01:50:08.391371 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.\n",
      "\n",
      "W0420 01:50:08.594653 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:973: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.\n",
      "\n",
      "/usr/local/lib/python3.7/site-packages/keras/engine/training_generator.py:47: UserWarning: Using a generator with `use_multiprocessing=True` and multiple workers may duplicate your data. Please consider using the`keras.utils.Sequence class.\n",
      "  UserWarning('Using a generator with `use_multiprocessing=True`'\n",
      "W0420 01:50:10.211843 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/callbacks.py:850: The name tf.summary.merge_all is deprecated. Please use tf.compat.v1.summary.merge_all instead.\n",
      "\n",
      "W0420 01:50:10.212799 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/callbacks.py:853: The name tf.summary.FileWriter is deprecated. Please use tf.compat.v1.summary.FileWriter instead.\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/60\n",
      "5/5 [==============================] - 47s 9s/step - loss: 5.4687 - rpn_class_loss: 0.0764 - rpn_bbox_loss: 2.0240 - mrcnn_class_loss: 1.4132 - mrcnn_bbox_loss: 0.8198 - mrcnn_mask_loss: 1.1352 - val_loss: 4.0008 - val_rpn_class_loss: 0.0343 - val_rpn_bbox_loss: 1.3574 - val_mrcnn_class_loss: 0.9520 - val_mrcnn_bbox_loss: 0.6232 - val_mrcnn_mask_loss: 1.0339\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "W0420 01:50:59.110243 4521070016 module_wrapper.py:139] From /usr/local/lib/python3.7/site-packages/keras/callbacks.py:995: The name tf.Summary is deprecated. Please use tf.compat.v1.Summary instead.\n",
      "\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 2/60\n",
      "5/5 [==============================] - 38s 8s/step - loss: 3.5603 - rpn_class_loss: 0.0589 - rpn_bbox_loss: 1.3560 - mrcnn_class_loss: 0.7596 - mrcnn_bbox_loss: 0.5318 - mrcnn_mask_loss: 0.8541 - val_loss: 2.8299 - val_rpn_class_loss: 0.0227 - val_rpn_bbox_loss: 0.7732 - val_mrcnn_class_loss: 0.7157 - val_mrcnn_bbox_loss: 0.6045 - val_mrcnn_mask_loss: 0.7138\n",
      "Epoch 3/60\n",
      "5/5 [==============================] - 39s 8s/step - loss: 2.4875 - rpn_class_loss: 0.0404 - rpn_bbox_loss: 0.7078 - mrcnn_class_loss: 0.5735 - mrcnn_bbox_loss: 0.4628 - mrcnn_mask_loss: 0.7030 - val_loss: 2.2913 - val_rpn_class_loss: 0.0199 - val_rpn_bbox_loss: 0.7623 - val_mrcnn_class_loss: 0.3917 - val_mrcnn_bbox_loss: 0.4791 - val_mrcnn_mask_loss: 0.6383\n",
      "Epoch 4/60\n",
      "5/5 [==============================] - 38s 8s/step - loss: 1.9112 - rpn_class_loss: 0.0256 - rpn_bbox_loss: 0.6571 - mrcnn_class_loss: 0.2530 - mrcnn_bbox_loss: 0.3179 - mrcnn_mask_loss: 0.6575 - val_loss: 2.3310 - val_rpn_class_loss: 0.0274 - val_rpn_bbox_loss: 1.2229 - val_mrcnn_class_loss: 0.2312 - val_mrcnn_bbox_loss: 0.2300 - val_mrcnn_mask_loss: 0.6195\n",
      "Epoch 5/60\n",
      "5/5 [==============================] - 38s 8s/step - loss: 1.5546 - rpn_class_loss: 0.0318 - rpn_bbox_loss: 0.5036 - mrcnn_class_loss: 0.1910 - mrcnn_bbox_loss: 0.2307 - mrcnn_mask_loss: 0.5975 - val_loss: 1.5934 - val_rpn_class_loss: 0.0189 - val_rpn_bbox_loss: 0.6750 - val_mrcnn_class_loss: 0.1243 - val_mrcnn_bbox_loss: 0.2093 - val_mrcnn_mask_loss: 0.5658\n",
      "Epoch 6/60\n",
      "5/5 [==============================] - 38s 8s/step - loss: 1.4191 - rpn_class_loss: 0.0221 - rpn_bbox_loss: 0.4265 - mrcnn_class_loss: 0.1901 - mrcnn_bbox_loss: 0.2118 - mrcnn_mask_loss: 0.5686 - val_loss: 1.7381 - val_rpn_class_loss: 0.0091 - val_rpn_bbox_loss: 0.7083 - val_mrcnn_class_loss: 0.2076 - val_mrcnn_bbox_loss: 0.2794 - val_mrcnn_mask_loss: 0.5337\n",
      "Epoch 7/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 1.6181 - rpn_class_loss: 0.0207 - rpn_bbox_loss: 0.7126 - mrcnn_class_loss: 0.1707 - mrcnn_bbox_loss: 0.1976 - mrcnn_mask_loss: 0.5165 - val_loss: 1.6919 - val_rpn_class_loss: 0.0220 - val_rpn_bbox_loss: 0.7192 - val_mrcnn_class_loss: 0.2082 - val_mrcnn_bbox_loss: 0.2484 - val_mrcnn_mask_loss: 0.4941\n",
      "Epoch 8/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 1.2464 - rpn_class_loss: 0.0157 - rpn_bbox_loss: 0.4490 - mrcnn_class_loss: 0.1283 - mrcnn_bbox_loss: 0.2058 - mrcnn_mask_loss: 0.4475 - val_loss: 1.5745 - val_rpn_class_loss: 0.0126 - val_rpn_bbox_loss: 0.6928 - val_mrcnn_class_loss: 0.1512 - val_mrcnn_bbox_loss: 0.2705 - val_mrcnn_mask_loss: 0.4474\n",
      "Epoch 9/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.8375 - rpn_class_loss: 0.0099 - rpn_bbox_loss: 0.1339 - mrcnn_class_loss: 0.0889 - mrcnn_bbox_loss: 0.1928 - mrcnn_mask_loss: 0.4120 - val_loss: 1.3867 - val_rpn_class_loss: 0.0097 - val_rpn_bbox_loss: 0.6513 - val_mrcnn_class_loss: 0.1272 - val_mrcnn_bbox_loss: 0.1922 - val_mrcnn_mask_loss: 0.4063\n",
      "Epoch 10/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.9882 - rpn_class_loss: 0.0203 - rpn_bbox_loss: 0.3128 - mrcnn_class_loss: 0.1452 - mrcnn_bbox_loss: 0.1527 - mrcnn_mask_loss: 0.3573 - val_loss: 1.3894 - val_rpn_class_loss: 0.0069 - val_rpn_bbox_loss: 0.6654 - val_mrcnn_class_loss: 0.1921 - val_mrcnn_bbox_loss: 0.1839 - val_mrcnn_mask_loss: 0.3412\n",
      "Epoch 11/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.7266 - rpn_class_loss: 0.0161 - rpn_bbox_loss: 0.1308 - mrcnn_class_loss: 0.0807 - mrcnn_bbox_loss: 0.1684 - mrcnn_mask_loss: 0.3305 - val_loss: 1.2448 - val_rpn_class_loss: 0.0103 - val_rpn_bbox_loss: 0.7417 - val_mrcnn_class_loss: 0.0491 - val_mrcnn_bbox_loss: 0.1596 - val_mrcnn_mask_loss: 0.2841\n",
      "Epoch 12/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 1.7710 - rpn_class_loss: 0.0398 - rpn_bbox_loss: 1.2163 - mrcnn_class_loss: 0.0886 - mrcnn_bbox_loss: 0.1379 - mrcnn_mask_loss: 0.2885 - val_loss: 1.3314 - val_rpn_class_loss: 0.0324 - val_rpn_bbox_loss: 0.7413 - val_mrcnn_class_loss: 0.1213 - val_mrcnn_bbox_loss: 0.1910 - val_mrcnn_mask_loss: 0.2454\n",
      "Epoch 13/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.5890 - rpn_class_loss: 0.0154 - rpn_bbox_loss: 0.1593 - mrcnn_class_loss: 0.0652 - mrcnn_bbox_loss: 0.1113 - mrcnn_mask_loss: 0.2377 - val_loss: 1.1368 - val_rpn_class_loss: 0.0085 - val_rpn_bbox_loss: 0.7001 - val_mrcnn_class_loss: 0.0616 - val_mrcnn_bbox_loss: 0.1351 - val_mrcnn_mask_loss: 0.2316\n",
      "Epoch 14/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.6807 - rpn_class_loss: 0.0088 - rpn_bbox_loss: 0.1454 - mrcnn_class_loss: 0.1609 - mrcnn_bbox_loss: 0.1226 - mrcnn_mask_loss: 0.2431 - val_loss: 1.1137 - val_rpn_class_loss: 0.0136 - val_rpn_bbox_loss: 0.6747 - val_mrcnn_class_loss: 0.0784 - val_mrcnn_bbox_loss: 0.1245 - val_mrcnn_mask_loss: 0.2226\n",
      "Epoch 15/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.5736 - rpn_class_loss: 0.0085 - rpn_bbox_loss: 0.1515 - mrcnn_class_loss: 0.0754 - mrcnn_bbox_loss: 0.1200 - mrcnn_mask_loss: 0.2181 - val_loss: 1.2982 - val_rpn_class_loss: 0.0248 - val_rpn_bbox_loss: 0.6824 - val_mrcnn_class_loss: 0.1905 - val_mrcnn_bbox_loss: 0.1589 - val_mrcnn_mask_loss: 0.2416\n",
      "Epoch 16/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.4706 - rpn_class_loss: 0.0135 - rpn_bbox_loss: 0.1003 - mrcnn_class_loss: 0.0605 - mrcnn_bbox_loss: 0.0841 - mrcnn_mask_loss: 0.2122 - val_loss: 1.1570 - val_rpn_class_loss: 0.0183 - val_rpn_bbox_loss: 0.6763 - val_mrcnn_class_loss: 0.0907 - val_mrcnn_bbox_loss: 0.1653 - val_mrcnn_mask_loss: 0.2064\n",
      "Epoch 17/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.3990 - rpn_class_loss: 0.0149 - rpn_bbox_loss: 0.0734 - mrcnn_class_loss: 0.0422 - mrcnn_bbox_loss: 0.0927 - mrcnn_mask_loss: 0.1759 - val_loss: 1.1502 - val_rpn_class_loss: 0.0150 - val_rpn_bbox_loss: 0.6836 - val_mrcnn_class_loss: 0.1182 - val_mrcnn_bbox_loss: 0.1537 - val_mrcnn_mask_loss: 0.1797\n",
      "Epoch 18/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.6277 - rpn_class_loss: 0.0214 - rpn_bbox_loss: 0.1893 - mrcnn_class_loss: 0.1490 - mrcnn_bbox_loss: 0.0982 - mrcnn_mask_loss: 0.1698 - val_loss: 1.0978 - val_rpn_class_loss: 0.0140 - val_rpn_bbox_loss: 0.7187 - val_mrcnn_class_loss: 0.0935 - val_mrcnn_bbox_loss: 0.0710 - val_mrcnn_mask_loss: 0.2005\n",
      "Epoch 19/60\n",
      "5/5 [==============================] - 34s 7s/step - loss: 0.6727 - rpn_class_loss: 0.0222 - rpn_bbox_loss: 0.2231 - mrcnn_class_loss: 0.1609 - mrcnn_bbox_loss: 0.0825 - mrcnn_mask_loss: 0.1842 - val_loss: 1.1734 - val_rpn_class_loss: 0.0100 - val_rpn_bbox_loss: 0.7386 - val_mrcnn_class_loss: 0.0882 - val_mrcnn_bbox_loss: 0.1626 - val_mrcnn_mask_loss: 0.1740\n",
      "Epoch 20/60\n",
      "5/5 [==============================] - 34s 7s/step - loss: 0.4185 - rpn_class_loss: 0.0174 - rpn_bbox_loss: 0.0979 - mrcnn_class_loss: 0.0482 - mrcnn_bbox_loss: 0.0922 - mrcnn_mask_loss: 0.1628 - val_loss: 1.2045 - val_rpn_class_loss: 0.0130 - val_rpn_bbox_loss: 0.6982 - val_mrcnn_class_loss: 0.2314 - val_mrcnn_bbox_loss: 0.0940 - val_mrcnn_mask_loss: 0.1678\n",
      "Epoch 21/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.4629 - rpn_class_loss: 0.0164 - rpn_bbox_loss: 0.1262 - mrcnn_class_loss: 0.0611 - mrcnn_bbox_loss: 0.1031 - mrcnn_mask_loss: 0.1561 - val_loss: 0.9847 - val_rpn_class_loss: 0.0096 - val_rpn_bbox_loss: 0.7061 - val_mrcnn_class_loss: 0.0445 - val_mrcnn_bbox_loss: 0.0860 - val_mrcnn_mask_loss: 0.1386\n",
      "Epoch 22/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.4306 - rpn_class_loss: 0.0154 - rpn_bbox_loss: 0.0956 - mrcnn_class_loss: 0.1037 - mrcnn_bbox_loss: 0.0855 - mrcnn_mask_loss: 0.1303 - val_loss: 1.0165 - val_rpn_class_loss: 0.0082 - val_rpn_bbox_loss: 0.6853 - val_mrcnn_class_loss: 0.0517 - val_mrcnn_bbox_loss: 0.1166 - val_mrcnn_mask_loss: 0.1547\n",
      "Epoch 23/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.3003 - rpn_class_loss: 0.0138 - rpn_bbox_loss: 0.0438 - mrcnn_class_loss: 0.0405 - mrcnn_bbox_loss: 0.0707 - mrcnn_mask_loss: 0.1315 - val_loss: 0.9904 - val_rpn_class_loss: 0.0108 - val_rpn_bbox_loss: 0.6625 - val_mrcnn_class_loss: 0.0673 - val_mrcnn_bbox_loss: 0.0737 - val_mrcnn_mask_loss: 0.1761\n",
      "Epoch 24/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.4676 - rpn_class_loss: 0.0185 - rpn_bbox_loss: 0.0650 - mrcnn_class_loss: 0.0804 - mrcnn_bbox_loss: 0.1582 - mrcnn_mask_loss: 0.1455 - val_loss: 1.1406 - val_rpn_class_loss: 0.0073 - val_rpn_bbox_loss: 0.6920 - val_mrcnn_class_loss: 0.0831 - val_mrcnn_bbox_loss: 0.1771 - val_mrcnn_mask_loss: 0.1812\n",
      "Epoch 25/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.2980 - rpn_class_loss: 0.0086 - rpn_bbox_loss: 0.0189 - mrcnn_class_loss: 0.0447 - mrcnn_bbox_loss: 0.0996 - mrcnn_mask_loss: 0.1263 - val_loss: 1.1090 - val_rpn_class_loss: 0.0124 - val_rpn_bbox_loss: 0.6919 - val_mrcnn_class_loss: 0.0794 - val_mrcnn_bbox_loss: 0.1617 - val_mrcnn_mask_loss: 0.1636\n",
      "Epoch 26/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 1.1547 - rpn_class_loss: 0.0348 - rpn_bbox_loss: 0.7685 - mrcnn_class_loss: 0.0587 - mrcnn_bbox_loss: 0.1714 - mrcnn_mask_loss: 0.1214 - val_loss: 1.0618 - val_rpn_class_loss: 0.0190 - val_rpn_bbox_loss: 0.6979 - val_mrcnn_class_loss: 0.0751 - val_mrcnn_bbox_loss: 0.1254 - val_mrcnn_mask_loss: 0.1443\n",
      "Epoch 27/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 1.1745 - rpn_class_loss: 0.0342 - rpn_bbox_loss: 0.7875 - mrcnn_class_loss: 0.0637 - mrcnn_bbox_loss: 0.1417 - mrcnn_mask_loss: 0.1474 - val_loss: 1.1630 - val_rpn_class_loss: 0.0170 - val_rpn_bbox_loss: 0.7096 - val_mrcnn_class_loss: 0.0695 - val_mrcnn_bbox_loss: 0.1940 - val_mrcnn_mask_loss: 0.1730\n",
      "Epoch 28/60\n",
      "5/5 [==============================] - 35s 7s/step - loss: 0.2340 - rpn_class_loss: 0.0106 - rpn_bbox_loss: 0.0409 - mrcnn_class_loss: 0.0216 - mrcnn_bbox_loss: 0.0483 - mrcnn_mask_loss: 0.1125 - val_loss: 1.0031 - val_rpn_class_loss: 0.0102 - val_rpn_bbox_loss: 0.7041 - val_mrcnn_class_loss: 0.0702 - val_mrcnn_bbox_loss: 0.0933 - val_mrcnn_mask_loss: 0.1252\n",
      "Epoch 29/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.3014 - rpn_class_loss: 0.0161 - rpn_bbox_loss: 0.0733 - mrcnn_class_loss: 0.0285 - mrcnn_bbox_loss: 0.0738 - mrcnn_mask_loss: 0.1097 - val_loss: 0.9929 - val_rpn_class_loss: 0.0190 - val_rpn_bbox_loss: 0.6854 - val_mrcnn_class_loss: 0.0747 - val_mrcnn_bbox_loss: 0.0660 - val_mrcnn_mask_loss: 0.1478\n",
      "Epoch 30/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.4114 - rpn_class_loss: 0.0084 - rpn_bbox_loss: 0.1104 - mrcnn_class_loss: 0.0540 - mrcnn_bbox_loss: 0.1091 - mrcnn_mask_loss: 0.1295 - val_loss: 1.0322 - val_rpn_class_loss: 0.0076 - val_rpn_bbox_loss: 0.6628 - val_mrcnn_class_loss: 0.0676 - val_mrcnn_bbox_loss: 0.1102 - val_mrcnn_mask_loss: 0.1841\n",
      "Epoch 31/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.5729 - rpn_class_loss: 0.0098 - rpn_bbox_loss: 0.2394 - mrcnn_class_loss: 0.1079 - mrcnn_bbox_loss: 0.0832 - mrcnn_mask_loss: 0.1327 - val_loss: 1.1332 - val_rpn_class_loss: 0.0086 - val_rpn_bbox_loss: 0.6743 - val_mrcnn_class_loss: 0.1072 - val_mrcnn_bbox_loss: 0.1207 - val_mrcnn_mask_loss: 0.2224\n",
      "Epoch 32/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.6189 - rpn_class_loss: 0.0132 - rpn_bbox_loss: 0.3095 - mrcnn_class_loss: 0.0612 - mrcnn_bbox_loss: 0.1052 - mrcnn_mask_loss: 0.1298 - val_loss: 1.1758 - val_rpn_class_loss: 0.0120 - val_rpn_bbox_loss: 0.7839 - val_mrcnn_class_loss: 0.0579 - val_mrcnn_bbox_loss: 0.1236 - val_mrcnn_mask_loss: 0.1982\n",
      "Epoch 33/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.4078 - rpn_class_loss: 0.0088 - rpn_bbox_loss: 0.1050 - mrcnn_class_loss: 0.0796 - mrcnn_bbox_loss: 0.1028 - mrcnn_mask_loss: 0.1116 - val_loss: 1.3312 - val_rpn_class_loss: 0.0163 - val_rpn_bbox_loss: 0.8286 - val_mrcnn_class_loss: 0.1812 - val_mrcnn_bbox_loss: 0.1453 - val_mrcnn_mask_loss: 0.1599\n",
      "Epoch 34/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.4323 - rpn_class_loss: 0.0152 - rpn_bbox_loss: 0.1286 - mrcnn_class_loss: 0.0643 - mrcnn_bbox_loss: 0.0932 - mrcnn_mask_loss: 0.1311 - val_loss: 1.3567 - val_rpn_class_loss: 0.0145 - val_rpn_bbox_loss: 0.7106 - val_mrcnn_class_loss: 0.3632 - val_mrcnn_bbox_loss: 0.1253 - val_mrcnn_mask_loss: 0.1430\n",
      "Epoch 35/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.4198 - rpn_class_loss: 0.0102 - rpn_bbox_loss: 0.0652 - mrcnn_class_loss: 0.0706 - mrcnn_bbox_loss: 0.0998 - mrcnn_mask_loss: 0.1740 - val_loss: 1.0425 - val_rpn_class_loss: 0.0115 - val_rpn_bbox_loss: 0.6677 - val_mrcnn_class_loss: 0.1182 - val_mrcnn_bbox_loss: 0.0888 - val_mrcnn_mask_loss: 0.1563\n",
      "Epoch 36/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.5697 - rpn_class_loss: 0.0139 - rpn_bbox_loss: 0.2317 - mrcnn_class_loss: 0.1109 - mrcnn_bbox_loss: 0.0859 - mrcnn_mask_loss: 0.1272 - val_loss: 1.0238 - val_rpn_class_loss: 0.0087 - val_rpn_bbox_loss: 0.6700 - val_mrcnn_class_loss: 0.0662 - val_mrcnn_bbox_loss: 0.1063 - val_mrcnn_mask_loss: 0.1726\n",
      "Epoch 37/60\n",
      "5/5 [==============================] - 34s 7s/step - loss: 0.5158 - rpn_class_loss: 0.0131 - rpn_bbox_loss: 0.2681 - mrcnn_class_loss: 0.0439 - mrcnn_bbox_loss: 0.0861 - mrcnn_mask_loss: 0.1047 - val_loss: 1.1210 - val_rpn_class_loss: 0.0163 - val_rpn_bbox_loss: 0.6927 - val_mrcnn_class_loss: 0.0880 - val_mrcnn_bbox_loss: 0.1496 - val_mrcnn_mask_loss: 0.1745\n",
      "Epoch 38/60\n",
      "5/5 [==============================] - 34s 7s/step - loss: 0.3877 - rpn_class_loss: 0.0203 - rpn_bbox_loss: 0.1096 - mrcnn_class_loss: 0.0549 - mrcnn_bbox_loss: 0.1004 - mrcnn_mask_loss: 0.1024 - val_loss: 1.1125 - val_rpn_class_loss: 0.0126 - val_rpn_bbox_loss: 0.6524 - val_mrcnn_class_loss: 0.0968 - val_mrcnn_bbox_loss: 0.1498 - val_mrcnn_mask_loss: 0.2008\n",
      "Epoch 39/60\n",
      "5/5 [==============================] - 35s 7s/step - loss: 0.4165 - rpn_class_loss: 0.0097 - rpn_bbox_loss: 0.1299 - mrcnn_class_loss: 0.0942 - mrcnn_bbox_loss: 0.0799 - mrcnn_mask_loss: 0.1027 - val_loss: 1.0497 - val_rpn_class_loss: 0.0141 - val_rpn_bbox_loss: 0.6377 - val_mrcnn_class_loss: 0.0790 - val_mrcnn_bbox_loss: 0.1349 - val_mrcnn_mask_loss: 0.1841\n",
      "Epoch 40/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.3431 - rpn_class_loss: 0.0160 - rpn_bbox_loss: 0.1422 - mrcnn_class_loss: 0.0176 - mrcnn_bbox_loss: 0.0653 - mrcnn_mask_loss: 0.1020 - val_loss: 0.9856 - val_rpn_class_loss: 0.0134 - val_rpn_bbox_loss: 0.6787 - val_mrcnn_class_loss: 0.0880 - val_mrcnn_bbox_loss: 0.0557 - val_mrcnn_mask_loss: 0.1500\n",
      "Epoch 41/60\n",
      "5/5 [==============================] - 39s 8s/step - loss: 0.2478 - rpn_class_loss: 0.0135 - rpn_bbox_loss: 0.0373 - mrcnn_class_loss: 0.0452 - mrcnn_bbox_loss: 0.0451 - mrcnn_mask_loss: 0.1066 - val_loss: 1.1825 - val_rpn_class_loss: 0.0082 - val_rpn_bbox_loss: 0.7667 - val_mrcnn_class_loss: 0.0817 - val_mrcnn_bbox_loss: 0.1291 - val_mrcnn_mask_loss: 0.1968\n",
      "Epoch 42/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.3082 - rpn_class_loss: 0.0139 - rpn_bbox_loss: 0.0536 - mrcnn_class_loss: 0.0408 - mrcnn_bbox_loss: 0.0745 - mrcnn_mask_loss: 0.1253 - val_loss: 1.1787 - val_rpn_class_loss: 0.0081 - val_rpn_bbox_loss: 0.6948 - val_mrcnn_class_loss: 0.1352 - val_mrcnn_bbox_loss: 0.1175 - val_mrcnn_mask_loss: 0.2231\n",
      "Epoch 43/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.3127 - rpn_class_loss: 0.0130 - rpn_bbox_loss: 0.0378 - mrcnn_class_loss: 0.0511 - mrcnn_bbox_loss: 0.0752 - mrcnn_mask_loss: 0.1356 - val_loss: 1.1727 - val_rpn_class_loss: 0.0119 - val_rpn_bbox_loss: 0.6492 - val_mrcnn_class_loss: 0.1016 - val_mrcnn_bbox_loss: 0.1219 - val_mrcnn_mask_loss: 0.2881\n",
      "Epoch 44/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.2055 - rpn_class_loss: 0.0142 - rpn_bbox_loss: 0.0188 - mrcnn_class_loss: 0.0237 - mrcnn_bbox_loss: 0.0433 - mrcnn_mask_loss: 0.1055 - val_loss: 1.0293 - val_rpn_class_loss: 0.0080 - val_rpn_bbox_loss: 0.6168 - val_mrcnn_class_loss: 0.0863 - val_mrcnn_bbox_loss: 0.0961 - val_mrcnn_mask_loss: 0.2220\n",
      "Epoch 45/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.2564 - rpn_class_loss: 0.0092 - rpn_bbox_loss: 0.0303 - mrcnn_class_loss: 0.0436 - mrcnn_bbox_loss: 0.0698 - mrcnn_mask_loss: 0.1035 - val_loss: 1.1607 - val_rpn_class_loss: 0.0069 - val_rpn_bbox_loss: 0.6704 - val_mrcnn_class_loss: 0.1633 - val_mrcnn_bbox_loss: 0.1107 - val_mrcnn_mask_loss: 0.2095\n",
      "Epoch 46/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.2919 - rpn_class_loss: 0.0080 - rpn_bbox_loss: 0.0556 - mrcnn_class_loss: 0.0760 - mrcnn_bbox_loss: 0.0629 - mrcnn_mask_loss: 0.0893 - val_loss: 1.3460 - val_rpn_class_loss: 0.0165 - val_rpn_bbox_loss: 0.8707 - val_mrcnn_class_loss: 0.0669 - val_mrcnn_bbox_loss: 0.1557 - val_mrcnn_mask_loss: 0.2362\n",
      "Epoch 47/60\n",
      "5/5 [==============================] - 37s 7s/step - loss: 0.4582 - rpn_class_loss: 0.0073 - rpn_bbox_loss: 0.1637 - mrcnn_class_loss: 0.0930 - mrcnn_bbox_loss: 0.0942 - mrcnn_mask_loss: 0.1001 - val_loss: 1.2854 - val_rpn_class_loss: 0.0165 - val_rpn_bbox_loss: 0.9023 - val_mrcnn_class_loss: 0.0926 - val_mrcnn_bbox_loss: 0.0765 - val_mrcnn_mask_loss: 0.1976\n",
      "Epoch 48/60\n",
      "5/5 [==============================] - 50s 10s/step - loss: 0.2973 - rpn_class_loss: 0.0187 - rpn_bbox_loss: 0.0528 - mrcnn_class_loss: 0.0218 - mrcnn_bbox_loss: 0.0959 - mrcnn_mask_loss: 0.1081 - val_loss: 0.9407 - val_rpn_class_loss: 0.0103 - val_rpn_bbox_loss: 0.6456 - val_mrcnn_class_loss: 0.0791 - val_mrcnn_bbox_loss: 0.0773 - val_mrcnn_mask_loss: 0.1283\n",
      "Epoch 49/60\n",
      "5/5 [==============================] - 40s 8s/step - loss: 0.3996 - rpn_class_loss: 0.0055 - rpn_bbox_loss: 0.1244 - mrcnn_class_loss: 0.0483 - mrcnn_bbox_loss: 0.1074 - mrcnn_mask_loss: 0.1141 - val_loss: 0.9699 - val_rpn_class_loss: 0.0159 - val_rpn_bbox_loss: 0.6625 - val_mrcnn_class_loss: 0.0696 - val_mrcnn_bbox_loss: 0.1048 - val_mrcnn_mask_loss: 0.1171\n",
      "Epoch 50/60\n",
      "5/5 [==============================] - 37s 7s/step - loss: 0.5507 - rpn_class_loss: 0.0175 - rpn_bbox_loss: 0.3054 - mrcnn_class_loss: 0.0813 - mrcnn_bbox_loss: 0.0622 - mrcnn_mask_loss: 0.0843 - val_loss: 1.0845 - val_rpn_class_loss: 0.0103 - val_rpn_bbox_loss: 0.7285 - val_mrcnn_class_loss: 0.0817 - val_mrcnn_bbox_loss: 0.0890 - val_mrcnn_mask_loss: 0.1750\n",
      "Epoch 51/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.3978 - rpn_class_loss: 0.0159 - rpn_bbox_loss: 0.1690 - mrcnn_class_loss: 0.0730 - mrcnn_bbox_loss: 0.0535 - mrcnn_mask_loss: 0.0864 - val_loss: 1.1801 - val_rpn_class_loss: 0.0194 - val_rpn_bbox_loss: 0.7624 - val_mrcnn_class_loss: 0.1164 - val_mrcnn_bbox_loss: 0.1147 - val_mrcnn_mask_loss: 0.1673\n",
      "Epoch 52/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.3026 - rpn_class_loss: 0.0132 - rpn_bbox_loss: 0.0658 - mrcnn_class_loss: 0.0542 - mrcnn_bbox_loss: 0.0631 - mrcnn_mask_loss: 0.1063 - val_loss: 0.9829 - val_rpn_class_loss: 0.0087 - val_rpn_bbox_loss: 0.6877 - val_mrcnn_class_loss: 0.0938 - val_mrcnn_bbox_loss: 0.0617 - val_mrcnn_mask_loss: 0.1310\n",
      "Epoch 53/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.4454 - rpn_class_loss: 0.0086 - rpn_bbox_loss: 0.2137 - mrcnn_class_loss: 0.0467 - mrcnn_bbox_loss: 0.0806 - mrcnn_mask_loss: 0.0957 - val_loss: 1.0042 - val_rpn_class_loss: 0.0034 - val_rpn_bbox_loss: 0.6690 - val_mrcnn_class_loss: 0.0845 - val_mrcnn_bbox_loss: 0.0746 - val_mrcnn_mask_loss: 0.1727\n",
      "Epoch 54/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.2722 - rpn_class_loss: 0.0108 - rpn_bbox_loss: 0.0171 - mrcnn_class_loss: 0.0670 - mrcnn_bbox_loss: 0.0685 - mrcnn_mask_loss: 0.1089 - val_loss: 1.0614 - val_rpn_class_loss: 0.0095 - val_rpn_bbox_loss: 0.6706 - val_mrcnn_class_loss: 0.0824 - val_mrcnn_bbox_loss: 0.1329 - val_mrcnn_mask_loss: 0.1661\n",
      "Epoch 55/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.2834 - rpn_class_loss: 0.0112 - rpn_bbox_loss: 0.0572 - mrcnn_class_loss: 0.0624 - mrcnn_bbox_loss: 0.0625 - mrcnn_mask_loss: 0.0901 - val_loss: 1.5297 - val_rpn_class_loss: 0.0236 - val_rpn_bbox_loss: 1.0357 - val_mrcnn_class_loss: 0.2001 - val_mrcnn_bbox_loss: 0.1120 - val_mrcnn_mask_loss: 0.1584\n",
      "Epoch 56/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.2222 - rpn_class_loss: 0.0068 - rpn_bbox_loss: 0.0512 - mrcnn_class_loss: 0.0436 - mrcnn_bbox_loss: 0.0469 - mrcnn_mask_loss: 0.0737 - val_loss: 1.1819 - val_rpn_class_loss: 0.0127 - val_rpn_bbox_loss: 0.7409 - val_mrcnn_class_loss: 0.1118 - val_mrcnn_bbox_loss: 0.1073 - val_mrcnn_mask_loss: 0.2092\n",
      "Epoch 57/60\n",
      "5/5 [==============================] - 33s 7s/step - loss: 0.2878 - rpn_class_loss: 0.0114 - rpn_bbox_loss: 0.0741 - mrcnn_class_loss: 0.0440 - mrcnn_bbox_loss: 0.0712 - mrcnn_mask_loss: 0.0871 - val_loss: 1.0394 - val_rpn_class_loss: 0.0291 - val_rpn_bbox_loss: 0.6530 - val_mrcnn_class_loss: 0.0883 - val_mrcnn_bbox_loss: 0.1105 - val_mrcnn_mask_loss: 0.1585\n",
      "Epoch 58/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.3093 - rpn_class_loss: 0.0094 - rpn_bbox_loss: 0.0687 - mrcnn_class_loss: 0.0768 - mrcnn_bbox_loss: 0.0593 - mrcnn_mask_loss: 0.0950 - val_loss: 1.0129 - val_rpn_class_loss: 0.0106 - val_rpn_bbox_loss: 0.6551 - val_mrcnn_class_loss: 0.0909 - val_mrcnn_bbox_loss: 0.0897 - val_mrcnn_mask_loss: 0.1665\n",
      "Epoch 59/60\n",
      "5/5 [==============================] - 32s 6s/step - loss: 0.3363 - rpn_class_loss: 0.0093 - rpn_bbox_loss: 0.1339 - mrcnn_class_loss: 0.0444 - mrcnn_bbox_loss: 0.0623 - mrcnn_mask_loss: 0.0865 - val_loss: 1.2361 - val_rpn_class_loss: 0.0121 - val_rpn_bbox_loss: 0.6844 - val_mrcnn_class_loss: 0.2530 - val_mrcnn_bbox_loss: 0.1330 - val_mrcnn_mask_loss: 0.1536\n",
      "Epoch 60/60\n",
      "5/5 [==============================] - 37s 7s/step - loss: 0.4788 - rpn_class_loss: 0.0094 - rpn_bbox_loss: 0.2251 - mrcnn_class_loss: 0.0412 - mrcnn_bbox_loss: 0.1033 - mrcnn_mask_loss: 0.0999 - val_loss: 0.9376 - val_rpn_class_loss: 0.0115 - val_rpn_bbox_loss: 0.6983 - val_mrcnn_class_loss: 0.0535 - val_mrcnn_bbox_loss: 0.0568 - val_mrcnn_mask_loss: 0.1175\n"
     ]
    }
   ],
   "source": [
    "Cell.train(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
