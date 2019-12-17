# Real-time Smile Recognition
The goal of this project is to develop a real time smile detection algorithm running on OpenCV using deep learning in keras

## Installation

### Step 1

Install Python 3.6

### Step 2

Navigate to the root directory (where this README is located) and run

```bash
pip install -e .
```

### Step 3

Download [genki4k dataset](https://inc.ucsd.edu/mplab/wordpress/wp-content/uploads/genki4k.tar). And put it into "project-root/data/" folder.

## Running the project

Train the model for the first time from root directory with

```bash
python src/train.py
```

Using the same command again does not train the model again unless the relevant
line in the code is changed.

Run the camera script with

```bash
python src/run_camera.py
```

and press `q` to quit.

Or run one image though the model with

```bash
python src/run.py
```

Change the file path in the code to change the image.

## Directory structure

- /
    - requirements.txt: the project requirements
    - setup.py: automatic installation with `pip install . -e`
- /src/
    - models/: directory for different kinds of neural models
    - utils/: directory for helper functions
    - run_camera.py: running the real-time application for smile detection
    - run.py: run only once on a static file (no evaluation)
    - train.py: train and evaluate the model
- /data/
    - genki4k/: contains the extracted dataset as downloaded from [http://mplab.ucsd.edu/wordpress/?page_id=398]
    - models/: directory for the saved trained models
- /report/: Contains the report files
