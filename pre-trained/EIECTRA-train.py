#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time
time_start=time.time()
print("***************************the program starts***************************")

from my_tokenizers2 import SmilesTokenizer
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tokenizer = SmilesTokenizer(vocab_file ='mol2vec_vocabs.txt')

# !rm -rf smiles_tokenizer
# tokenizer.save_pretrained("smiles_tokenizer")


# In[ ]:


# import pandas as pd
# e15_file = '/data/deepdta/data/E15.csv'
# e15_data = pd.read_csv(e15_file,header=None)
# e15_data.head()


# In[ ]:


# e15_data.columns = ['dataset', 'id', 'smiles']


# In[ ]:


# e15_smiles_list = e15_data['smiles'].to_list()


# In[ ]:


# with open("/data/deepdta/data/e15_smile_train.txt", "w") as outfile:
#     outfile.write("\n".join(e15_smiles_list))


# In[ ]:


# import deepsmiles
# print("DeepSMILES version: %s" % deepsmiles.__version__)
# converter = deepsmiles.Converter(rings=True, branches=True)
# print(converter) # record the options used

# encoded = converter.encode("c1cccc(C(=O)Cl)c1")
# print("Encoded: %s" % encoded)


# In[2]:


from transformers import ElectraModel, ElectraConfig,ElectraForPreTraining,ElectraForMaskedLM
# from simpletransformers.language_modeling import LanguageModelingModel
emb_dim = 256
configuration = ElectraConfig(vocab_size=3357, hidden_size=emb_dim)
configuration.output_hidden_states=True
configuration.output_attentions=True
configuration.max_position_embeddings = 256

# electra_model = ElectraForPreTraining.from_pretrained('outputs/best_model/')
electra_model = ElectraForMaskedLM(configuration)
# electra_model = ElectraForMaskedLM.from_pretrained('newsmiles_output/')
 
electra_model.num_parameters()


# In[3]:


# input_ids = tokenizer("ON(CCCP(O)(O)=O)C=O", return_tensors="pt")["input_ids"]
# outputs = electra_model(input_ids, labels=input_ids)
# print(outputs[2][0].size())

input_str = '864942730 2228262129 999602603 1175930089 2245384272 1842269008 2246703798 530246988 2246703798 4110198278 1016841875 195319295 2245384272'

input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"]
outputs = electra_model(input_ids, labels=input_ids)
print(outputs[2][0].size())


# In[4]:


input_ids


# In[ ]:


# with open('../Data/smile_train.txt', 'r') as rfile:
#     train_smiles = rfile.readlines()

# with open('../Data/deepsmile_train.txt', 'w') as wfile:
#     for line in train_smiles:
#         encoded = converter.encode(line)
#         wfile.writelines(encoded)

        


# In[ ]:


# with open('../Data/smile_test.txt', 'r') as rfile:
#     train_smiles = rfile.readlines()

# with open('../Data/deepsmile_test.txt', 'w') as wfile:
#     for line in train_smiles:
#         encoded = converter.encode(line)
#         wfile.writelines(encoded)

        


# In[5]:


from transformers import LineByLineTextDataset
from transformers import ElectraTokenizer
# tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="mol2vec_corpus_e15_small.txt",
    block_size=128,
)
# eval_dataset = LineByLineTextDataset(
#     tokenizer=tokenizer,
#     file_path="../Data/smile_test.txt",
#     block_size=128,
# )


# In[6]:


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


# In[18]:


from transformers import Trainer, TrainingArguments

output_dir="fingerprints_smile_output"+str(emb_dim)


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_gpu_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)


# In[19]:


trainer = Trainer(
    model= electra_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
#     eval_dataset = eval_dataset,
    prediction_loss_only=True
)


# In[20]:


#get_ipython().run_cell_magic('time', '', 'trainer.train()')
trainer.train()

# In[21]:


trainer.save_model(output_dir)

print("***************************the program ends***************************")
time_end=time.time()
print('totally cost:',time_end-time_start)

"""
input_ids = tokenizer("ON(CCCP(O)(O)=O)C=O", return_tensors="pt")["input_ids"]
electra_model = ElectraForMaskedLM.from_pretrained('smiles_output')
outputs = electra_model(input_ids, labels=input_ids)


# In[ ]:


outputs[2][0].size() 


# In[ ]:


from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=70,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


# In[ ]:


from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)


# In[ ]:


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)


# In[ ]:


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'trainer.train()')


# In[ ]:



"""
