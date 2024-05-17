from flask import Flask, request, jsonify
from rnn import LOTTO_RNN
from mlp import LOTTO_MLP
from rf import LOTTO_RANDOMFOREST
from transformer import LottoTransformer
from cnn import LOTTO_CNN
import pickle

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello World!'

@app.route('/hello', methods=['POST'])
def hello_world2():
    return 'Hello World!'

@app.route('/predict', methods=['GET'])
def hello_w():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # RNN
            # 모델 파일 로드
            model_rnn = pickle.load(open('model_RNN.pkl', 'rb'))
            # Make prediction
            rnn_prediction = model_rnn.predict()

            # MLP
            model_mlp = pickle.load(open('model_MLP.pkl', 'rb'))
            mlp_prediction = model_mlp.predict()

            # Random Forest
            model_rf = pickle.load(open('model_RANDOMFOREST.pkl', 'rb'))
            rf_prediction = model_rf.predict()

            # Transformer
            model_trans = pickle.load(open('model_TRANSFORMER.pkl', 'rb'))
            trans_prediction = model_trans.predict()

            # CNN
            model_cnn = pickle.load(open('model_CNN.pkl', 'rb'))
            cnn_prediction = model_cnn.predict()

            # # PCA
            # model_pca = pickle.load(open('model_PCA.pkl', 'rb'))
            # pca_prediction = model_pca.predict()

            
            return jsonify({'rnn_prediction': str(rnn_prediction),'mlp_prediction': str(mlp_prediction),
                            'rf_prediction': str(rf_prediction), 'trans_prediction': str(trans_prediction),'cnn_prediction': str(cnn_prediction)})
        except Exception as e:
            return jsonify({'error': str(e), 'message': 'Error processing request'})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=3000, debug=True)
