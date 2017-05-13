This is a solution to the hard version of IPSC 2016 Problem L
(https://ipsc.ksp.sk/2016/real/problems/l.html).

Project structure:

* test/*.png
  800 images of size 600 x 70 px, each represents six hieroglyphs.

* test/sample.out
  First 200 lines of the file are translations for the corresponding
  images, other lines don't have any sense.
  
* prepare-training-data.py
  Splits first tagged 200 images from the test directory by letters,
  simplifies them to black & white images of size 18 x 18 px and puts
  into corresponding sub-directories in the train directory.

* train-model.py
  Takes tagged small images from the train directory and trains simple
  ANN, then saves ANN to the file.

* solver.py
  Takes paths to the test directory and model, predicts translations
  for all images in the test directory.


Typical usage:

$ ./prepare-training-data.py --test_dir test --train_dir train
$ ./train-model.py --train_dir data --model_path model.h5
$ ./solver.py --test_dir test --model_path model.h5


Requirements:

Python3 and following modules: pil, keras (works with theano or google
tensorflow), sklearn, h5py, numpy.
