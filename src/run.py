"""Static run functions"""
from src.train import load


def load_detection_model():
    model_fn = 'data/models/net_l2.json'
    weights_fn = 'data/models/weights_l2.h5'
    model = load(model_fn, weights_fn)
    return model
