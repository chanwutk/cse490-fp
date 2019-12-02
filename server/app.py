import os
import re

from src.utils import to_layer_info, tensor_to_base64s, weights_to_base64s
from src.visualizable_net import TraceableAlexNet, TraceableVgg
from src.data_loaders import load_base64_image, load_classes
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(
    app,
    resources={
        r"/classify": {"origins": "*"},
        r"/network-layout": {"origins": "*"},
        r"/trace": {"origins": "*"},
    },
)

cwd = os.getcwd()
weights_path = os.path.join(cwd, "data/weights.pt")

USE_ALEX = True

if USE_ALEX:
    model = TraceableAlexNet(num_classes=5, traceable=True)
    model.load_model(weights_path)
    class_names = load_classes("class_names.txt")
else:
    model = TraceableVgg(traceable=True, pretrained=True)
    class_names = load_classes("class_names_imagenet.txt")


@app.route("/classify", methods=["POST"])
@cross_origin(origin="localhost", headers=["Content-Type"])
def classify():
    if request.method != "POST":
        return "/classify only accept POST request", 500

    data = re.sub("^data:image/.+;base64,", "", request.json["data"])
    image = load_base64_image(data, do_normalize=USE_ALEX)
    output = model.classify(image)
    class_idx = output.view(-1).max(0)[1]
    return class_names[class_idx.item()]


@app.route("/trace/<int:layer_idx>", methods=["GET"])
@cross_origin(origin="localhost", headers=["Content-Type"])
def get_vis(layer_idx: int):
    traces = model.get_traces()
    if len(traces) <= layer_idx:
        return "index out of bound for traces", 500

    input_tensor, output_tensor, layer, weights = traces[layer_idx]
    output = {
        "input": tensor_to_base64s(input_tensor),
        "output": tensor_to_base64s(output_tensor),
    }
    if weights is not None:
        output["weights"] = weights_to_base64s(weights)

    return jsonify(output)


@app.route("/network-layout", methods=["GET"])
@cross_origin(origin="localhost", headers=["Content-Type"])
def network_layout():
    return jsonify(list(map(to_layer_info, model.get_traces())))


if __name__ == "__main__":
    app.run(port=5432)
