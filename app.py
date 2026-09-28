from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model = YOLO("best.pt")


def reconstruct_word(result, model):
    boxes = result.boxes.xyxy.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy()
    class_names = model.names
    
    chars = []
    for i, box in enumerate(boxes):
        xmin = box[0]
        char = class_names[int(labels[i])]
        chars.append((xmin, char))
    
    chars = sorted(chars, key=lambda x: x[0])
    word = "".join([c[1] for c in chars])
    return word

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        
        file = request.files["file"]
        if file.filename == "":
            return "No file selected"
        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        
        results = model.predict(filepath, conf=0.25)

        words = []
        for r in results:
            word = reconstruct_word(r, model)
            words.append(word)

        return render_template("index.html", filename=file.filename, words=words)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
