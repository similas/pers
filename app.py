from ml.text_classifier import TextClassifier
from emotion_mapper import EmotionMapper


if __name__=="__main__":
    text = "i'm so tired i want to fuck someone"
    emotion = TextClassifier().inference(text)
    EmotionMapper().find_perfume(emotion)
