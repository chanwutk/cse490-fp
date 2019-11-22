from src.visualizable_net import VisualizableAlexNet
from flask import Flask

app = Flask(__name__)
model = VisualizableAlexNet(num_classes=5, visualizable=True)
model.load_model("./data/weights.pt")


@app.route("/classify", methods=["POST"])
def classify():
    if request.method == "POST":
        # TODO: read the file from the url;
        # classify in model;
        # return text result
        pass


@app.route("/get-vis/<int:layer>")
def get_vis(layer):
    i, o, layer, w = model.get_visualizations()[layer]
    return layer


if __name__ == "__main__":
    app.run()
