from helper_functions import write_tupi
import transformers

#write_tupi(write_corpus=True)

tokenizer = transformers.BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased", do_lower_case=False)

print(tokenizer.tokenize("kɨgomət"))