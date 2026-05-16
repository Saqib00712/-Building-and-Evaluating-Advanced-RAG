#!/usr/bin/env python
# coding: utf-8

# # Lesson 2: RAG Triad of metrics

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()


# In[3]:


from trulens_eval import Tru

tru = Tru()
tru.reset_database()


# In[4]:


from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()


# In[5]:


from llama_index import Document

document = Document(text="\n\n".\
                    join([doc.text for doc in documents]))


# In[6]:


from utils import build_sentence_window_index

from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)


# In[7]:


from utils import get_sentence_window_query_engine

sentence_window_engine = \
get_sentence_window_query_engine(sentence_index)


# In[8]:


output = sentence_window_engine.query(
    "How do you create your AI portfolio?")
output.response


# ## Feedback functions

# In[9]:


import nest_asyncio

nest_asyncio.apply()


# In[10]:


from trulens_eval import OpenAI as fOpenAI

provider = fOpenAI()


# ### 1. Answer Relevance

# In[11]:


from trulens_eval import Feedback

f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()


# ### 2. Context Relevance

# In[12]:


from trulens_eval import TruLlama

context_selection = TruLlama.select_source_nodes().node.text


# In[13]:


import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)


# In[14]:


import numpy as np

f_qs_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)


# ### 3. Groundedness

# In[15]:


from trulens_eval.feedback import Groundedness

grounded = Groundedness(groundedness_provider=provider)


# In[16]:


f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons,
             name="Groundedness"
            )
    .on(context_selection)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)


# ## Evaluation of the RAG application

# In[17]:


from trulens_eval import TruLlama
from trulens_eval import FeedbackMode

tru_recorder = TruLlama(
    sentence_window_engine,
    app_id="App_1",
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)


# In[18]:


eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        eval_questions.append(item)


# In[19]:


eval_questions


# In[20]:


eval_questions.append("How can I be successful in AI?")


# In[21]:


eval_questions


# In[22]:


for question in eval_questions:
    with tru_recorder as recording:
        sentence_window_engine.query(question)


# In[23]:


records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()


# In[24]:


import pandas as pd

pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]


# In[25]:


tru.get_leaderboard(app_ids=[])


# In[26]:


tru.run_dashboard()

