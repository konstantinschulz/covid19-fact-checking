import os
import streamlit as st
import streamlit.components.v1 as components
from transformers import DistilBertForSequenceClassification, DistilBertConfig, Trainer, DistilBertTokenizer
import requests
import json
import pandas as pd
from spacy.lang.en import English
import docx2txt

from fact_checking_app.classes import CovidDataset

nlpSpacy = English()
nlpSpacy.add_pipe('sentencizer')
nlpSpacy.Defaults.stop_words |= {"we", "a", "the", "this"}
all_stopwords = nlpSpacy.Defaults.stop_words
# set your Google API key here
api_key = ""

# run npm run build in the frontend directory and set _RELEASE to true, if you do not plan to change the frontend component
_RELEASE = False

# Declare a Streamlit component. `declare_component` returns a function
# that is used to create instances of the component. We're naming this
# function "_component_func", with an underscore prefix, because we don't want
# to expose it directly to users. Instead, we will create a custom wrapper
# function, below, that will serve as our component's public API.

# It's worth noting that this call to `declare_component` is the
# *only thing* you need to do to create the binding between Streamlit and
# your component frontend. Everything else we do in this file is simply a
# best practice.

if not _RELEASE:
    _component_func = components.declare_component(
        "vue_component",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/dist")
    _component_func = components.declare_component(
        "vue_component", path=build_dir)


# Create a wrapper function for the component. This is an optional
# best practice - we could simply expose the component function returned by
# `declare_component` and call it done. The wrapper allows us to customize
# our component's API: we can pre-process its input args, post-process its
# output value, and add a docstring for users.


def vue_component(sentences, labels, key=None):
    component_value = _component_func(
        sentences=sentences, labels=labels, key=key, default=0)
    return component_value


docx_file = st.file_uploader("Upload your document", type=['txt', 'docx'])
st.write('Or')
sentences = st.text_area("Type in your text")

# read the text document into the sentences variable
if docx_file is not None:
    if docx_file.type == "text/plain":
        sentences = str(docx_file.read(), "utf-8")
    elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Parse in the uploadFile Class directory
        sentences = docx2txt.process(docx_file)

data = []  # an array of text+label pairs, since we do not know the true labels of each sentence, we just set the labels to 0

finalSentences = []  # an array of the sentences, used later in the frontend component
finalResults = []  # an array of the final results - 0 for regular sentences, for suspicious sentences - 1, if no relevant claims are found or an array of relevant claims, if there are any
if sentences != '':
    # parse text into sentences
    for sent in nlpSpacy(sentences).sents:
        data.append([sent.text, 0])
        finalSentences.append(sent.text)
    # save sentences and placeholder labels to dataframe
    df = pd.DataFrame(data, columns=['text', 'labels'])
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # tokenize sentences
    test_encodings = tokenizer(
        df.text.values.tolist(), truncation=True, padding=True)
    # create dataset
    X_test = CovidDataset(test_encodings, df.labels.values.tolist())
    # fetch fine-tuned model and config
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    config = DistilBertConfig.from_json_file(
        os.path.join(parent_dir, 'model/config.json'))
    model = DistilBertForSequenceClassification.from_pretrained(
        os.path.join(parent_dir, 'model'), config=config)
    trainer = Trainer(
        model=model,
    )
    # make predictions
    results = trainer.predict(test_dataset=X_test).predictions.argmax(-1)
    for i, result in enumerate(results):
        if (result == 0 and df.text.values.tolist()[i] != ''):
            finalResults.append(int(result))
        # for suspicious sentences: simplify sentence and send GET request to Google API
        elif (df.text.values.tolist()[i] != '' and result == 1):
            text_tokens = nlpSpacy.tokenizer(df.text.values.tolist()[i])
            tokens_without_punct = [
                word for word in text_tokens if not word.is_punct]
            tokens = [i.text for i in tokens_without_punct]
            tokens_without_sw = [
                word for word in tokens if not word in all_stopwords]
            sent = '%20'.join(tokens_without_sw)
            query = "https://factchecktools.googleapis.com/v1alpha1/claims:search?query={}&key={}".format(
                sent, api_key)
            r = requests.get(query)
            # if there are relevant claims found, save them in finalResults
            if json.loads(r.text) and 'claims' in json.loads(r.text):
                finalResults.append(json.loads(r.text)['claims'])
            else:
                finalResults.append(int(result))

st.markdown("---")

# We use the special "key" argument to assign a fixed identity to this
# component instance. By default, when a component's arguments change,
# it is considered a new instance and will be re-mounted on the frontend
# and lose its current state. In this case, we want to vary the component's
# "arguments without having it get recreated.
frontend_component = vue_component(sentences=finalSentences,
                                   labels=finalResults, key="foo")
