import nltk
import streamlit as st 

from transformers import pipeline
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

import spacy
from spacy import displacy
nlp = spacy.blank("en")

nltk.download('punkt')

# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen

# Pick model
model_name = "google/pegasus-xsum"
# Load pretrained tokenizer
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)

# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

def pegasus_summarizer(docx):
    # Define PEGASUS model
  pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)
  # Create tokens
  tokens = pegasus_tokenizer(docx, truncation=True, padding="longest", return_tensors="pt")
  
  # Generate the summary
  encoded_summary = pegasus_model.generate(**tokens)
  
  # Decode the summarized text
  decoded_summary = pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)
  
  # Print the summary
  print('Decoded Summary :',decoded_summary)
  
  summarizer = pipeline(
      "summarization", 
      model=model_name, 
      tokenizer=pegasus_tokenizer, 
      framework="pt"
  )
  
  summary = summarizer(docx, min_length=30, max_length=150)
  return summary[0]["summary_text"]

# Fetch Text From Url
@st.cache
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text


@st.cache(allow_output_mutation=True)
def analyze_text(text):
	return nlp(text)

def main():
	"""Summarizer Streamlit App"""

	st.title("Summarizer")

	activities = ["Summarize"]
	choice = st.sidebar.selectbox("Select Activity",activities)

	if choice == 'Summarize':
		st.subheader("Summarize Document")
		raw_text = st.text_area("Enter Text Here","Type Here")
		summarizer_type = st.selectbox("Summarizer Type", ["Pegasus", "Sumy Lex Rank"])
		if st.button("Summarize"):
			if summarizer_type == "Pegasus":
				summary_result = pegasus_summarizer(raw_text)
			elif summarizer_type == "Sumy Lex Rank":
				summary_result = sumy_summarizer(raw_text)

			st.write(summary_result)
				
		

if __name__ == '__main__':
	main()



HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
