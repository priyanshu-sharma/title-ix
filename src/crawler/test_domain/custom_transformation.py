from llama_index.schema import TransformComponent
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
                'compound': vader_score.get('compound', None)
            }
        return nodes