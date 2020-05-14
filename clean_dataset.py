import os
import pandas as pd

data_deceptive = []
for i in range(1,6):
    directory = 'deceptive_from_MTurk/fold'+str(i)
    for filename in os.listdir(directory):
        f = open(directory+"/"+filename, "r")
        lines = f.read()
        data_deceptive.append([lines, "-1.0", i])
    directory = 'truthful_from_TripAdvisor/fold' + str(i)
    for filename in os.listdir(directory):
        f = open(directory + "/" + filename, "r")
        lines = f.read()
        data_deceptive.append([lines, "1.0", i])

df = pd.DataFrame(data_deceptive, columns=["Sentence", "Label", "Fold"])
df.to_csv("final_data.csv", index=False)
