import os
import logging
import regex
import re
import pickle
import pandas as pd
import ast
from vllm import LLM, SamplingParams

def predict_llm(arg_model):
    if arg_model:
        model_id = arg_model
    else:
        model_id ="mistralai/Mistral-7B-Instruct-v0.2"

    logger = logging.getLogger(__name__)
    os.environ['HF_HOME'] = '/workspace/cache/'



    llm = LLM(model=model_id, 
              max_num_seqs=60,
             tensor_parallel_size=1, 
             dtype="float16")

    def template_func(address):

        template = f"""<s>[INST]
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
        {{"Name": "[0]-THOMASSEN [1]-GULBRANDSEN [2]-OG [3]-GUNDERSEN", "StreetName": "[5]-TV [6]-SD [7]-9", "Municipality": "[8]-JAPARATINGA", "PostalCode": "[9]-57950 [10]-000", "CountryCode": "[11]-BR"}}


        ### INPUT:
        {address}
        [/INST]

        ### OUTPUT:
        """
        return template

    stop = ["\n\n",
            "\n \n",
            "\n  \n",
            "\n   \n",
            "\n    \n",
            "\n     \n",
            "\n      \n",
            "\n       \n"]

    sampling_params = SamplingParams(temperature=0.2, top_p=0.5, max_tokens=3000, stop=stop)

    sample_size = 1000

    # INPUT
    df = pickle.load(open("data/V2_test.pkl", 'rb'))[:sample_size]

    # Add index and create prompts
    df["full_address"] = df.apply(lambda row: " ".join("[" + str(i) + "]-" + token for i, token in enumerate(row["sentence"])), axis=1)
    df = df.reset_index(drop=True)
    full_addresses = df["full_address"].to_dict()
    full_addresses = [template_func(full_addresses[i]) for i in range(len(full_addresses))]

    # Using vLLM to do inference
    outputs = llm.generate(full_addresses[:1000], sampling_params)


    def parse_output(out):
      pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
      # cleanup last character
      out = re.sub(r"]$", "}", out)
      out_match = pattern.findall(out)
      if out_match:
          try:
              # cleanup some hallucinations
              return ast.literal_eval(out_match[0].replace(': "Cannot be classified as any of the given classes"', "")
                                      .replace(": true", "")
                                      .replace(': "_gaito_spa_ar"', "")
                                      .replace("{{", "{")
                                      .replace("}}", "}")
                                     )
          except:
              logger.info("No json found in:" + out)
              return {}
      else:
        logger.info("Wrong format:" + out)
        return {}


    def flatten_json(nested_json):
        flattened_json = {}

        def flatten(x, name=''):
          if type(x) is dict:
             for a in x:
                flatten(x[a], a)
          else:
             flattened_json[name] = x

        flatten(nested_json)
        return flattened_json


    def get_llm_tags(outputs_llm):
      """
      Create the list of tags corresponding to the dictionary of the form {"tag": ["[i]-word"]}

      Args:
          outputs_llm: dictionary of the form {"tag": ["[i]-word", ...]}

      Returns:
          List of tags corresponding to each words in the same order as "sentence".
      """
      llm_tag_list = []
      for i, output_llm in enumerate(outputs_llm):
        data_tags = df.loc[i, "tags"]
        data_words = df.loc[i, "sentence"]
        llm_tags = ["OOA" for tag in data_tags]
        for k, v in output_llm.items():
          if v and isinstance(v, str):
            pattern = regex.compile(r"(?<=\[)([0-9]*?)(?=\])")
            word_pattern = regex.compile(r"(?<=\]-).*")
            for word in v.split(" "):
              id = pattern.findall(word)
              s = word_pattern.findall(word)
              if s:
                idxs = [i for i, x in enumerate(data_words) if x == s[0]]
              else:
                idxs = [i for i, x in enumerate(data_words) if x == word]
                if len(idxs) == 0:
                  logger.info(word + " not in input: " + str(data_words))
                  continue
                if len(idxs) == 1:
                  llm_tags[idxs[0]] = k
                if len(idxs) > 1:
                  llm_tags[idxs[0]] = k
              if id:
                idx = int(id[0])
                if idx in idxs:
                  if word[-1] == "$":
                    llm_tags[idx] = "HardSep"
                  else:
                    llm_tags[idx] = k
                else:
                  if len(idxs) == 1:
                    logger.info(f"{str(data_words)}: Label modified for {word}, setting it as {idxs[0]}")
                    llm_tags[idxs[0]] = k
                  else:
                    logger.info("There is an ambiguity for:" + str(s) + " in " + str(data_words) + " keeping OOA")
              else:
                if len(idxs) == 1:
                  llm_tags[idxs[0]] = k
                else:
                  logger.info("There is an ambiguity for:" + str(s) + " in " + str(data_words) + " keeping OOA")
        llm_tag_list.append(llm_tags)
      return llm_tag_list


    # Parse LLM output
    outputs_llm = [flatten_json(parse_output(out.outputs[0].text)) for out in outputs]

    llm_tag_list = get_llm_tags(outputs_llm)

    # Compute F1 score

    df["llm_tags"] = pd.Series(llm_tag_list)
    pattern = regex.compile(r"(?<=-).*")
    df["label_tags"] = df.apply(lambda tags: [pattern.findall(tag)[0] for tag in tags["tags"]], axis = 1)

    df["precision"] = df.apply(lambda x: sum(c == cp for c, cp in zip(x["label_tags"], x["llm_tags"]) if c not in ["HardSep"]) / sum([c not in ["HardSep"] for c in x["label_tags"]]), axis=1)
    df["recall"] = df.apply(
        lambda x: sum(
            [
                c == cp != "OOA"
                for c, cp in zip(x["label_tags"], x["llm_tags"])
                if c not in ["SoftSep", "HardSep"]
            ]
            )
            / sum([c not in ["HardSep", "OOA"] for c in x["label_tags"]])
              if sum([c!="OOA" for c in x["label_tags"]]) !=0
              else 1,
              axis=1)
    df["f1"] = df.apply(lambda x: 2*x["precision"] * x["recall"] / (x["precision"] + x["recall"]) if (x["precision"] + x["recall"]) != 0 else 0, axis=1)

    logger.info("F1 score:", df["f1"].mean())

    df.to_pickle("path/to/results")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Specify model path")
    args = parser.parse_args()

    model = args.model  # access the argument value using its name
    predict_llm(model)