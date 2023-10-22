import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json
import random
import pickle
import numpy as np
import tkinter as tk
from tkinter import Scrollbar, Text, Entry, Button, END

# Load model and data
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict the intent class
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Function to get a response from the chatbot
def getResponse(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Function to send a message and get a response
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n', 'user')
        ChatLog.tag_config('user', foreground='black')  # Set user text color to black

        res = chatbot_response(msg)
        ChatLog.insert(tk.END, "Nyay_Rakshak: " + res + '\n\n', 'bot')
        ChatLog.tag_config('bot', foreground='black')  # Set bot text color to black

        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

# Create the main application window
base = tk.Tk()
base.title("Nyay_Rakshak")
base.geometry("400x500")
base.resizable(width=False, height=False)

# Create Chat window
ChatLog = Text(base, bd=0, bg="#f2f2f2", height="8", width="50", font="Helvetica", wrap=tk.WORD)
ChatLog.config(state=tk.DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#00BFFF", activebackground="#3c9d9b",foreground='black', command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="black", width="29", height="5", font="Helvetica")

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
