import os
import shutil
import re

from src.visualizable_net import TraceableAlexNet
from src.data_loaders import load_single_image, load_classes
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(
    app,
    resources={
        r"/classify": {"origins": "*"},
        r"/network-layout": {"origins": "*"},
        r"/trace/": {"origins": "*"},
    },
)

cwd = os.getcwd()
visualizations_dir = os.path.join(cwd, "data/__visualizations__")
weights_path = os.path.join(cwd, "data/weights.pt")

model = TraceableAlexNet(num_classes=5, traceable=True)
model.load_model(weights_path)
class_names = load_classes()


@app.route("/classify", methods=["POST"])
@cross_origin(origin="localhost", headers=["Content-Type"])
def classify():
    if request.method == "POST":
        data = re.sub("^data:image/.+;base64,", "", request.json["data"])
        image = load_single_image(data)
        output = model.classify(image)
        class_idx = output.view(-1).max(0)[1]
        return class_names[class_idx.item()]
    else:
        return None


@app.route("/trace/<int:layer>")
def get_vis(layer):
    return "hello" + str(layer)
    # i, o, layer, w = model.get_traces()[layer]
    # print(i.size())  # [1, dim, x, y]
    # print(o.size())  # [1, dim, x, y]
    # print(w.size())  # [in, out, k, k]
    # return layer


def to_layer_info(layer):
    layer_type = type(layer).__name__
    layer_info = {"type": layer_type}
    if layer_type == "Conv2d":
        layer_info["inputDim"] = layer.in_channels
        layer_info["outputDim"] = layer.out_channels
        layer_info["stride"] = layer.stride[0]
        layer_info["kernelSize"] = layer.kernel_size[0]
    elif layer_type == "MaxPool2d":
        layer_info["stride"] = layer.stride[0]
        layer_info["kernelSize"] = layer.kernel_size[0]
    return layer_info


@app.route("/network-layout", methods=["GET"])
@cross_origin(origin="localhost", headers=["Content-Type"])
def network_layout():
    return jsonify(list(map(to_layer_info, model.get_traces())))


if __name__ == "__main__":
    if os.path.exists(visualizations_dir):
        shutil.rmtree(visualizations_dir)
    os.mkdir(visualizations_dir)
    app.run(port=5432)
