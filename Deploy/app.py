from flask import Flask,render_template,request
import pickle
import numpy as np

#model = pickle.load(open('Roberta.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



import seaborn as sns
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# ummm="I'm so sad"
# def polarity_scores_roberta(umm):
#     encoded_text = tokenizer(umm, return_tensors='pt')
#     output = model(**encoded_text)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)
#     scores_dict = {
#         'negative' : scores[0]*100,
#         'neutral' : scores[1],
#         'positive' : scores[2]
#     }
#     return scores_dict

@app.route('/predict',methods=['post'])
def sentiment_Analysis():
    Input_Sentence = request.form.get('Sentence')
    encoded_text = tokenizer(Input_Sentence, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'negative' : scores[0]*100,
        'neutral' : scores[1]*100,
        'positive' : scores[2]*100
    }

    #result=model.predict(np.array(Input_Sentence))
    return str(scores_dict)




# @app.route('/predict',methods=['post'])
# def sentiment_Analysis():
#     Input_Sentence = request.form.get('Sentence')
#     result=model.predict(np.array(Input_Sentence))
#     return str(result)
    

if __name__ == '__main__':
    app.run(debug=True)