import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('wrd.pkl', 'rb'))
classes = pickle.load(open('cls.pkl', 'rb'))

def clean_up(sentence):
    sentence_wrds = nltk.word_tokenize(sentence)
    sentence_wrds = [lemmatizer.lemmatize(word.lower()) for word in sentence_wrds]
    return sentence_wrds

def bow(sentence, words, show_details=True):
    sentence_wrds = clean_up(sentence)
    bag = [0]*len(words)
    for s in sentence_wrds:
        for i, w in enumerate(words):
            if w==s:
                bag[i]=1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))
               
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r>ERROR_THESHOLD]
    results.sort(key=lambda x: x[1], everse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result

def bot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return bot_response

if __name__ == "main":
    app.run()
    