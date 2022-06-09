SSMI - Semantic Segmentation of Medical Images:

This code implements a modified U-Net architecture to perform semantic segmentation of ultrasound scans of the heart. The class labels are 0 - background, 1 - left ventricle, 2 - myocardium, and 3 - left atrium.



>>> Steps for downloading dataset:

1. Go to https://www.creatis.insa-lyon.fr/Challenge/camus/index.html

2. Navigate to the "Datasets" tab, and click the think "online evaluation website"

3. Make an account and download the CAMUS training and testing datasets



>>> Steps for running this code:

1. Open main.py and change dataset_path to point to a folder that contains "training" and "testing" folders that have the same structure as the CAMUS dataset.

2. Select which subsets of data you want to train with in line 32 of main.py. You can specify image quality, chamber views, and systole or diastole.

3. Pick which U-Net variant you would like to evaluate by looking at the .py folders in SSMI_final/model. Initialize the selected model, change line 96 in main.py.

4. open a terminal session and navigate to folder that contains main.py

5. run the command "python main.py"

6. After each epoch, the validation set is evaluated and the IOUs for each class are printed out.

7. Once the network has finished learning, all figures and outputs are written to SSMI_final/plots.

8. A copy of the best performing network (according to validation performance) will be saved in the same folder as main.py.
