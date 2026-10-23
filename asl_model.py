"""Load the repository's dense ASL model without a TensorFlow runtime."""
import json

import h5py
import numpy as np


class ASLModel:
    def __init__(self, path):
        self.layers = []
        with h5py.File(path, "r") as model_file:
            config = json.loads(model_file.attrs["model_config"])
            if config["class_name"] != "Sequential":
                raise ValueError("Expected a Sequential ASL model")
            input_size = 63
            for layer in config["config"]["layers"]:
                kind, options = layer["class_name"], layer["config"]
                # Dropout is disabled at inference time.
                if kind in ("InputLayer", "Dropout"):
                    continue
                if kind != "Dense" or options["activation"] not in ("relu", "softmax"):
                    raise ValueError(f"Unsupported inference layer: {kind}")
                group = model_file["model_weights"][options["name"]]
                weights = {}

                def read_weight(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weights[name.rsplit("/", 1)[-1].split(":")[0]] = np.asarray(obj, dtype=np.float32)

                group.visititems(read_weight)
                kernel, bias = weights["kernel"], weights["bias"]
                if kernel.shape != (input_size, options["units"]) or bias.shape != (options["units"],):
                    raise ValueError("Invalid ASL model weight dimensions")
                kernel.flags.writeable = bias.flags.writeable = False
                self.layers.append((kernel, bias, options["activation"]))
                input_size = options["units"]
            if input_size != 28 or not self.layers or self.layers[-1][2] != "softmax":
                raise ValueError("Expected 28 ASL class probabilities")

    def __call__(self, features):
        output = np.asarray(features, dtype=np.float32)
        if output.ndim != 2 or output.shape[1] != 63:
            raise ValueError("Expected a batch of 63 hand coordinates")
        for kernel, bias, activation in self.layers:
            output = output @ kernel + bias
            if activation == "relu":
                output = np.maximum(output, 0)
            else:
                output = np.exp(output - np.max(output, axis=1, keepdims=True))
                output /= np.sum(output, axis=1, keepdims=True)
        return output
