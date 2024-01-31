from llama_index.schema import TransformComponent
from textblob import TextBlob

class TextBlobTransformation(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node_data = TextBlob(node.text)
            node['textblob'] = {
                'polarity': node_data.sentiment.polarity,
                'subjectivity': node_data.sentiment.subjectivity
            }
        return nodes