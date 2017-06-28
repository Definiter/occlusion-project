# occlusion-project

- pbs\_script/: Scripts for running code on cluster with PBS.
- slurm\_script/: Scripts for running code on cluster with slurm.
- data/: Data folder, including flickr, imagenet and PASCAL 3D+. Included in .gitignore.
- result/: Result of the experiments. Included in .gitignore.
- src/: Source code for current project. (A.ipynb → B.py: using A to generate B, and running B on cluster with PBS / slurm scripts.)
    - Settings
        - settings.py 
        - constant.py
    - Prepare datasets
        - dataset.ipynb → dataset.py: Generate training dataset and test dataset. Basically first 3 cells are used. 
    - Finetune
        - finetune.ipynb → finetune.py: Create network files, finetune networks and visualize middle results.
    - Test
        - test\_lmdb.ipynb → test.py: Generate test\_{}\_{}.prototxt for testing, and test model by reading images from lmdb.
        - test.ipynb → test.py: Test model by reading images directly from the disk. [Deprecated]
        - data_utility.ipynb: Some utilities to show and test selected images in datasets. 
    - Analyse
        - img2vec.ipynb → img2vec.py: Extract feature vectors of images.
        - visualize\_finetune: Plot training loss, training accuracy and validation accuracy during finetuning. 
        - visualize\_accuracy: Plot accuracy curves according to different occlusion levels.
        - visualize\_img2vec: Plot feature space, and accuracy improvement for each class.
        - visualize\_aperture: Plot results of aperture occlusion. [Deprecated]
        - vec2accuracy\_divide: Test results with divided test datasets to get variation. [Deprecated]
        - heat\_map.ipynb: Plot confidence heat maps of given images. 
    - Miscellaneous
        - estimate\_time.ipynb
- legacy/: Legacy code related to visual concept, not used by current project.
    - legacy/src\_visual\_concept/: Use visual concept to generate occluded images.
    - legacy/src\_cl/: Chen Liu's code of Picasso project, modified by Hao Wang.
    - legacy/src\_orignal/: Chen Liu's original code of Picasso project.



