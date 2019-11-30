# Real-time Smile Recognition
The goal of this project is to develop a real time smile detection algorithm running on OpenCV using deep learning in keras

## Installation
Navigate to the root directory (where this README is located) and run
```
pip install -e .
```

## Running the project
Train the model for the first time from root directory with
```
python src/train.py
```

Using the same command again does not train the model again unless the relevant
line in the code is changed.

Run the camera script with
```
python src/run_camera.py
```
and press `q` to quit.

Or run one image though the model with
```
python src/run.py
```
Change the file path in the code to change the image.
