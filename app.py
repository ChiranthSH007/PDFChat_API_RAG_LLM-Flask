from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from RAGBot import RAGPDFBot

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


class ClientApp():

    # global previous_model, previous_threads, previous_max_tokens, previous_top_k, previous_file, previous_chunk_size, previous_overlap, previous_temp

    def __init__(self):
        self.RAGBot = RAGPDFBot()
        # Pre-set values
        self.previous_file = None
        self.previous_model = ""
        self.previous_top_k = 2
        self.previous_temp = 0.7
        self.previous_threads = 64
        self.previous_rag_off = False
        self.previous_chunk_size = 500
        self.previous_overlap = 50
        self.previous_max_token = 50
        self.model_loaded = False
        self.params_loaded = False


# @app.route("/", methods=['GET'])
# @cross_origin()
# def home():
#     return render_template('index.html')


@app.route("/loadmodel", methods=['POST'])
@cross_origin()
def model():
    # Eg: C:/Users/win10/Desktop/Materials/Deep Learning/Deep Learning Notes.pdf
    result = ""
    model = request.args.get('model')
    file_path = request.args.get('filepath')
    chunk_size = int(request.args.get('chunksize'))
    overlap = int(request.args.get('overlap'))
    if chatApp.previous_model != model:
        chatApp.RAGBot.get_model(model=model)
        chatApp.previous_model = model
        chatApp.model_loaded = True
        print("loaded model")
        result = "Loaded Model"
    if file_path != None:
        print("File found")
        if chatApp.previous_file != file_path or chatApp.previous_chunk_size != chunk_size or chatApp.previous_overlap != overlap:
            print(file_path)
            chatApp.RAGBot.build_vectordb(
                chunk_size=chunk_size, overlap=overlap, file_path=file_path)
            chatApp.previous_chunk_size = chunk_size
            chatApp.previous_overlap = overlap
            chatApp.previous_file = file_path
            result = "File Loaded"
    return result, 200


@app.route("/params", methods=['POST'])
@cross_origin()
def params():
    top_k = int(request.args.get('topk'))
    temp = float(request.args.get('temp'))
    threads = int(request.args.get('threads'))
    rag_off = bool(request.args.get('ragoff'))
    chunk_size = int(request.args.get('chunksize'))
    max_tokens = int(request.args.get('maxtoken'))
    # Check if the model is loaded
    if chatApp.model_loaded:
        if chatApp.previous_top_k != top_k or chatApp.previous_temp != temp or chatApp.previous_threads != threads or chatApp.previous_rag_off != rag_off or chatApp.previous_chunk_size != chunk_size or chatApp.previous_max_token != max_tokens:
            chatApp.RAGBot.load_model(repeat_penalty=1.50, top_k=top_k, temp=temp,
                                      n_threads=threads, n_batch=threads, max_tokens=max_tokens)
            chatApp.previous_top_k = top_k
            chatApp.previous_temp = temp
            chatApp.previous_threads = threads
            chatApp.previous_rag_off = rag_off
            chatApp.previous_chunk_size = chunk_size
            chatApp.previous_max_token = max_tokens
            chatApp.params_loaded = True
            return "Parameters Updated", 200
    else:
        # Pass a specific error and check for it in front end
        print("model not loaded... please load the model")
        return "Model not loaded ", 200


@app.route("/chat", methods=['POST'])
@cross_origin()
def chat():
    # model_dropdown, query_text, top_k_slider, rag_off_checkbox, chunk_size_input, overlap_input, dataset_dropdown, threads_slider, max_token_input, repeat_penalty_input, temp_slider
    query_text = request.args.get('querytext')
    rag_off = bool(request.args.get('ragoff'))
    if (chatApp.params_loaded and chatApp.model_loaded):
        chatApp.RAGBot.retrieval(user_input=query_text,
                                 top_k=chatApp.previous_top_k, rag_off=rag_off)
        response = chatApp.RAGBot.inference()
        return jsonify(response)
    else:
        # Pass a specific error and check for it in front end
        print("model and pramas not loaded... please load the model")
        return "Mmodel and pramas not loaded ", 200


if __name__ == "__main__":
    chatApp = ClientApp()
    app.run(host="0.0.0.0", port=8080)
