# SVHN - Deep Learning using Caffe

Steps:

1.  Run convertMATtoLMDB.py 
     Caffe prefers lmdb files for large datasets. Hence need to convert .MAT files to lmdb files. 
     We should now have train, test, and extra lmdb files in the working directory.
2.  Make sure the paths in solver and and train_test prototxt files are updated.
3.  Run model.py

We can also try changing the solver functions, number of layers, kernels etc to dwelve deep.

Help: If anyone can create a cumulative confusion matrix, please let me know.

