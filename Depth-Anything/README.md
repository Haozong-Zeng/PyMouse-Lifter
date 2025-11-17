# Depth-Anything based training / inference guide
Real Time 3-D Pose Estimation for Mice with Only 2-D Annotation Via Data Synthesis
## Fine-tune a Depth Anything model with automatically synthetic data
Go to metric_depth for instructions. Our fine-tuned models can be downloaded from the links in metric_depth/checkpoints/links_to_models.txt
## Process the fine-tuned models
Use LoadFine-tunedModel.ipynb
## Offline Demo
Download the data from the link in assets/PyMouseLifter_demo.txt and run run_PyMouseLifter_offline_demo.py
## Replicate the delay & FPS estimation with flashing LEDs
Use run_PyMouseLifter_online_classification_LED.py with devices like LabJack U3-LV, 2 LEDs with appropriate switching circuit, and a high-speed camera.
