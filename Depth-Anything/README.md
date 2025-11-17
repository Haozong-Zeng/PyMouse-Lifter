# Depth-Anything based training / inference guide
Real Time 3-D Pose Estimation for Mice with Only 2-D Annotation Via Data Synthesis
## Fine-tune a Depth Anything model with automatically synthetic data
Go to metric_depth for instructions. The name of the dataset is "PyMouse_HQ". You might need to modified the train/test input suffix (e.g. PyMouse_HQ_train_files_with_gt_orbbec_trans.txt means synthetic data, corresponding to real data captured by orbbec depth camera, with style-transfer; remove "_orbbec_trans" before training) for switching between synthetic data & real data. Our fine-tuned models can be downloaded from the links in metric_depth/checkpoints/links_to_models.txt
## Process the fine-tuned models
Use LoadFine-tunedModel.ipynb
## Offline Demo
Download the data from the link in assets/PyMouseLifter_demo.txt and run run_PyMouseLifter_offline_demo.py
## Replicate the delay & FPS estimation with flashing LEDs
Use run_PyMouseLifter_online_classification_LED.py with devices like LabJack U3-LV, 2 LEDs with appropriate switching circuit, and a high-speed camera.
