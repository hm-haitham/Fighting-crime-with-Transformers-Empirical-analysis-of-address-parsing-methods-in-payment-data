import os
import logging
import pickle
import torch
import transformers
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTTrainer
import argparse

def train_llm(arg_model):

    if arg_model:
        model_id = arg_model
    else:
        model_id ="mistralai/Mistral-7B-Instruct-v0.2"
    
    MODEL = model_path.split("/")[-1]
    print("Model to train is",MODEL)
    
    logger = logging.getLogger(__name__)
    os.environ['HF_HOME'] = '/workspace/cache/'

    # The input expects a column named "sentence" containing a list of strings corresponding to space-separated words
    # The input expects a column named "tags" with target fields (same length as "sentence")
    df = pickle.load(open("data/V2_train.pkl", 'rb'))

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_length=3000
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        is_split_into_words=True,
        padding_side="right"
    )

    template ="""<s>[INST]
    You are a word classifier that classifies words from a text corresponding to an address free text field.
    You should analyze with deep precision the INPUT and return a dictionary with the following keys: "Name", "StreetNumber", "StreetName", "Municipality", "PostalCode", "Unit", "Country", "CountryCode".
    Each word is separated by a space and should be classified without any modification.
    Each word in the input has a prefix with the index i as '[i]-' and it should be ignored for the classification but it should remain AS-IS in the output.
    Sub sequence of words should be classified as follow:
    'Name': words corresponding to an indiviual name or institution name.
    'StreetNumber': words corresponding to a street number.
    'StreetName': words corresponding to a street name.
    'Municipality': words corresponding to a municipality or city.
    'PostalCode': words corresponding to a postal code.
    'Unit': words corresponding to a unit number.
    'Country': words corresponding to a full country name.
    'CountryCode': words corresponding to a country iso2 code.

    Output Indicator:
    1. If you are not sure about one word, don't classify it.
    2. Usually a name comes before the address.
    3. "$" is indicating a large separator and it should not be classified.
    4. The output words should be taken from the input only and it should not be modified
    5. The same word cannot be used in two different classes.
    6. Words are classified subsequently.
    7. Empty classes should not appear in the output.
    8. Output should not include nested values.
    9. Each index are taken from the input itself and the index matches, e.g. the prefix '[i]-' remains unchanged for all words.

    For example:
    ### INPUT:
    "[0]-THOMASSEN [1]-GULBRANDSEN [2]-OG [3]-GUNDERSEN [4]-$ [5]-TV [6]-SD [7]-9 [8]-JAPARATINGA [9]-57950 [10]-000 [11]-BR"
    ### OUTPUT: 
    {"Name": "[0]-THOMASSEN [1]-GULBRANDSEN [2]-OG [3]-GUNDERSEN", "StreetName": "[5]-TV [6]-SD [7]-9", "Municipality": "[8]-JAPARATINGA", "PostalCode": "[9]-57950 [10]-000", "CountryCode": "[11]-BR"}


    ### INPUT:
    {full_address}
    [/INST]

    ### OUTPUT:
    """

    def create_output(row):
      d = defaultdict(str)
      for i, tag in enumerate(row["tags"]):
          try:
              d[tag[2:]] += " " + row["full_address"].split(" ")[i]
          except:
              logger.info(row)
              logger.info(d)
      return d

    # Concat prefix index
    df["full_address"] = df.apply(lambda row: " ".join("[" + str(i) + "]-" + token for i, token in enumerate(row["sentence"])), axis=1)
    # Create dictionary for target
    df["target_output"] = df.apply(create_output, axis=1)
    data = Dataset.from_pandas(df)

    # Create training prompt
    data = data.map(lambda example: {"prompt": f"{template}\n{example['target_output']}"})

    tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        dataset_text_field="prompt",
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            warmup_steps=0.03,
            max_steps=-1,
            learning_rate=2e-4,
            evaluation_strategy="steps",
            eval_steps=500,
            logging_steps=10,
            output_dir="outputs",
            optim="paged_adamw_32bit",  
            save_strategy="steps",
            log_level="info",
            logging_first_step=True
        ),
        max_seq_length=3000,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    new_model = "models" + model_id.split("/")[-1]
    trainer.model.save_pretrained(new_model)
    logger.info("Model saved in " + new_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Specify model path")
    args = parser.parse_args()

    model = args.model  # access the argument value using its name
    train_llm(model)
