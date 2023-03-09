from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from torch import argmax
from nltk.corpus import stopwords
from config import Config

class TextClassifier():

    def __init__(self) -> None:
        print("model is loading...")
        self.check_point = Config.TEXT_CLASSIFIER_MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.check_point)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.check_point)
        

    def inference(self, text: str)->None:
        text = self.preprocessing(text)
        tokens = self.tokenizer(text, padding=True, 
                                truncation=True,
                                return_tensors="pt",
                                max_length=Config.TOKENIZER_MAX_LENGTH)
        
        output = self.model(**tokens)
        label_num = argmax(output.logits)
        label_emotion = Config.labels_dict[label_num.item()]
        print(f"Emotion: {label_emotion}")

    def preprocessing(self, text:str) -> str:
        print("preprocessor is loading...")
        # nltk.download('stopwords')
        stopword = list(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in (stopword)])
        return text


