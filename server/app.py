import os
import shutil
from src.visualizable_net import VisualizableAlexNet
from src.data_loaders import load_single_image, load_classes
from flask import Flask


app = Flask(__name__)

cwd = os.getcwd()
visualizations_dir = os.path.join(cwd, "data/__visualizations__")
weights_path = os.path.join(cwd, "data/weights.pt")

model = VisualizableAlexNet(num_classes=5, visualizable=True)
model.load_model(weights_path)
class_names = load_classes()


@app.route("/classify/<path:image_path>")
def classify(image_path):
    # TODO: change to post
    image = load_single_image(image_path[1:])
    output = model.classify(image)
    class_idx = output.view(-1).max(0)[1]
    return class_names[class_idx.item()]


@app.route("/get-vis/<int:layer>")
def get_vis(layer):
    i, o, layer, w = model.get_visualizations()[layer]
    print(i.size())  # [1, dim, x, y]
    print(o.size())  # [1, dim, x, y]
    print(w.size())  # [in, out, k, k]
    return layer


if __name__ == "__main__":
    if os.path.exists(visualizations_dir):
        shutil.rmtree(visualizations_dir)
    os.mkdir(visualizations_dir)
    app.run()
