import json
from flask import Flask, request, abort

from model.Match import Match

# from flask_cors import CORS

app = Flask(__name__)


# CORS(app, origins="http://localhost:3001")

@app.route("/image-match", methods=['POST'])
def make_image_match():
    data = request.json
    match = Match(data['nameImage1'], data['nameImage2'],
                  data['keypoint'], data['descriptor'])

    print(match.__repr__())

    result = match.training()

    return result


if __name__ == "__main__":
    app.run()
