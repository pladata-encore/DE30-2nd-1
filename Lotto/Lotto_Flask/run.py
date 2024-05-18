from app import create_app

from app.models.rnn import LOTTO_RNN
from app.models.mlp import LOTTO_MLP
from app.models.rf import LOTTO_RANDOMFOREST
from app.models.transformer import LottoTransformer
from app.models.cnn import LOTTO_CNN

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
