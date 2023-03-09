
import pandas as pd
import json

class EmotionMapper():
    def __init__(self) -> None:
        pass

    def prepare_data(self) -> any:
        f = open ('data/note_probs.json', "r")
        data = json.loads(f.read())
        dataframe = pd.DataFrame.from_dict(data)
        return dataframe

    def find_perfume(self, emotion: str):
        df = self.prepare_data()
        # print(df.loc["caring"].sort_values(ascending=True))
        print(df.loc[emotion][:10])

