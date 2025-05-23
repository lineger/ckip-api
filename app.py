import os
import zipfile
import requests
from flask import Flask, request, jsonify
from ckiptagger import WS, POS, NER
from opencc import OpenCC

app = Flask(__name__)
MODEL_DIR = "./data"
MODEL_ZIP_URL = "https://github.com/lineger/ckip-api/releases/download/v1.0/ckip_data.zip"
MODEL_ZIP_PATH = "ckip_data.zip"

cc = OpenCC('t2s')

# 只解壓、避免佔記憶體初始化過早
if not os.path.exists(MODEL_DIR):
    print("[下載] 開始下載 Ckip 模型 zip...")
    r = requests.get(MODEL_ZIP_URL, stream=True)
    with open(MODEL_ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("[解壓] 解壓中...")
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    os.remove(MODEL_ZIP_PATH)
    print("[完成] 模型已下載並解壓至", MODEL_DIR)

ws, pos, ner = None, None, None

@app.before_first_request
def load_model():
    global ws, pos, ner
    print("[初始化] 載入模型中...")
    ws = WS(MODEL_DIR)
    pos = POS(MODEL_DIR)
    ner = NER(MODEL_DIR)
    print("[完成] 模型已初始化")

@app.route("/segment", methods=["POST"])
def segment():
    text = request.json.get("text", "")
    word_sentence_list = ws([text])
    return jsonify({"segment": word_sentence_list[0]})

@app.route("/pos", methods=["POST"])
def pos_analysis():
    text = request.json.get("text", "")
    word_sentence_list = ws([text])
    pos_sentence_list = pos(word_sentence_list)
    result = [(w, p) for w, p in zip(word_sentence_list[0], pos_sentence_list[0])]
    return jsonify({"pos": result})

@app.route("/ner", methods=["POST"])
def ner_analysis():
    text = request.json.get("text", "")
    word_sentence_list = ws([text])
    pos_sentence_list = pos(word_sentence_list)
    ner_sentence_list = ner(word_sentence_list, pos_sentence_list)
    return jsonify({"ner": ner_sentence_list[0]})

@app.route("/convert", methods=["POST"])
def convert():
    text = request.json.get("text", "")
    converted = cc.convert(text)
    return jsonify({"converted": converted})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
