import pandas as pd
import huggingface_hub
from datasets import load_dataset

huggingface_hub.login()

indo_data = load_dataset("openlanguagedata/flores_plus", "ind_Latn", split="devtest")
eng_data = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")


df = indo_data.to_pandas()
df2 = eng_data.to_pandas()


dataset = pd.DataFrame({
    "id": df["id"],
    "ind": df["text"],
    "eng": df2["text"]
})

dataset.to_csv("data/raw/flores_devtest.tsv", index=False)