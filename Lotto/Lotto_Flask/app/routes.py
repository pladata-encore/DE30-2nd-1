from flask import Blueprint, request, jsonify
import requests
import pickle

from app.models.rnn import LOTTO_RNN
from app.models.mlp import LOTTO_MLP
from app.models.rf import LOTTO_RANDOMFOREST
from app.models.transformer import LottoTransformer
from app.models.cnn import LOTTO_CNN

main = Blueprint('main', __name__)

@main.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello World!'

@main.route('/get_lottery_results', methods=['POST'])
def get_lottery_results():
    drwNo = 1119
    api_url = f'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={drwNo}'

    try:
        response = requests.get(api_url)
        data = response.json()
        
        winning_numbers = {
            'drawNo': data['drwNo'],
            'numbers': [data[f'drwtNo{i}'] for i in range(1, 7)],
            'bonusNumber': data['bnusNo']
        }

        return jsonify(winning_numbers), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            model_rnn = pickle.load(open('model_RNN.pkl', 'rb'))
            rnn_prediction = model_rnn.predict()

            model_mlp = pickle.load(open('model_MLP.pkl', 'rb'))
            mlp_prediction = model_mlp.predict()

            model_rf = pickle.load(open('model_RANDOMFOREST.pkl', 'rb'))
            rf_prediction = model_rf.predict()

            model_trans = pickle.load(open('model_TRANSFORMER.pkl', 'rb'))
            trans_prediction = model_trans.predict()

            model_cnn = pickle.load(open('model_CNN.pkl', 'rb'))
            cnn_prediction = model_cnn.predict()

            return jsonify({
                'rnn_prediction': str(rnn_prediction),
                'mlp_prediction': str(mlp_prediction),
                'rf_prediction': str(rf_prediction),
                'trans_prediction': str(trans_prediction),
                'cnn_prediction': str(cnn_prediction)
            })
        except Exception as e:
            return jsonify({'error': str(e), 'message': 'Error processing request'})
