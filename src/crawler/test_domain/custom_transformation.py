from llama_index.schema import TransformComponent
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

class TextBlobTransformation(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node_data = TextBlob(node.text)
            node.metadata['textblob'] = {
                'polarity': node_data.sentiment.polarity,
                'subjectivity': node_data.sentiment.subjectivity
            }
        return nodes

class VaderTransformation(TransformComponent):
    def __call__(self, nodes, **kwargs):
        sent_analyzer = SentimentIntensityAnalyzer()
        for node in nodes:
            node_data = sent_analyzer.polarity_scores(node.text)
            node.metadata['vader'] = {
                'negative': node_data.get('neg', None),
                'positive': node_data.get('pos', None),
                'neutral': node_data.get('neu', None),
                'compound': node_data.get('compound', None)
            }
        return nodes

class RobertaTranformation(TransformComponent):
    def __call__(self, nodes, **kwargs):
        roberta_pretrained_model = f"cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(roberta_pretrained_model)
        model = AutoModelForSequenceClassification.from_pretrained(roberta_pretrained_model)
        for node in nodes:
            roberta_encoded_text = tokenizer(node.text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
            output = model(**roberta_encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            node.metadata['roberta'] = {
                'negative': scores[0],
                'neutral': scores[1],
                'positive': scores[2],
            }
        return nodes

class BertTransformation(TransformComponent):
    def __call__(self, nodes, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
        for node in nodes:
            bert_encoded_text = tokenizer(node.text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
            output = model(**bert_encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            node.metadata['bert'] = {
                'left': scores[0],
                'center': scores[1],
                'right': scores[2],
            }
        return nodes