import pandas as pd

def save_history(history, save_path):

    df = pd.DataFrame(history)

    df.to_csv(save_path, index=False)