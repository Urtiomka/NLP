# NLP Document Summarizer

This is an app created for summarization of the text.
It is streamed to Streamlit and is used there as a ready-to-use application.

This application has two types of summarization to choose from:
1. hugging-face transformer(pegasus-xsum)
2. sumy (lex rank summarizer)

## How to use

1. Visiting Streamlit app:

[Link to Streamlit app](https://eketweaw7f6ebpmkaobcuc.streamlit.app/)

2. Cloning repository:

Clone the repository and run in `cmd` following command:
```bash
streamlit run NLP.py
```

## WARNING
The Pegasus choice probably won't work on Streamlit Cloud because of not enough resources. In order to try Pegasus you have clone repo and try it on your own machine.
