# read pickle and print
from pandas.plotting import table
import pandas as pd
import os

# read the cosine similarity dataframe
cosine_similarity_df = pd.read_pickle('./data/metrics/cosine_similarity_df.pkl')

# plot average, low and high cosine similarity scores for prompt 1 - model mb vs ft as a line plot
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="temp", y="cosine_similarity_avg", hue="model", data=cosine_similarity_df[cosine_similarity_df['prompt']=='prompt1'])
plt.title('Cosine Similarity Scores for Prompt 1 - Model Baseline vs Fine-tuned')
plt.show()


plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="temp", y="cosine_similarity_avg", hue="model", data=cosine_similarity_df[cosine_similarity_df['prompt']=='prompt2'])
plt.title('Cosine Similarity Scores for Prompt 2 - Model Baseline vs Fine-tuned')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="temp", y="cosine_similarity_avg", hue="model", data=cosine_similarity_df[cosine_similarity_df['prompt']=='prompt3'])
plt.title('Cosine Similarity Scores for Prompt 3 - Model Baseline vs Fine-tuned')
plt.show()

# plot average cosine similarity scores averaged over all prompts for model mb vs ft as a line plot
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="temp", y="cosine_similarity_avg", hue="model", data=cosine_similarity_df)
plt.title('Average Cosine Similarity Scores for Model Baseline vs Fine-tuned')
plt.show()



# read the bm25 dataframe
bm25_df = pd.read_pickle('./data/metrics/bm25_df.pkl')

# plot average, low and high bm25 scores for prompt 1 - model mb vs ft as a line plot
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="temp", y="bm25_avg_score", hue="model", data=bm25_df[bm25_df['prompt']=='prompt1'])
plt.title('BM25 Scores for Prompt 1 - Model Baseline vs Fine-tuned')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="temp", y="bm25_avg_score", hue="model", data=bm25_df[bm25_df['prompt']=='prompt2'])
plt.title('BM25 Scores for Prompt 2 - Model Baseline vs Fine-tuned')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="temp", y="bm25_avg_score", hue="model", data=bm25_df[bm25_df['prompt']=='prompt3'])
plt.title('BM25 Scores for Prompt 3 - Model Baseline vs Fine-tuned')
plt.show()

# plot average bm25 scores averaged over all prompts for model mb vs ft as a line plot
plt.figure(figsize=(10, 6))
ax = sns.lineplot(x="temp", y="bm25_avg_score", hue="model", data=bm25_df)
plt.title('Average BM25 Scores for Model Baseline vs Fine-tuned')
plt.show()



## concatenate the cosine similarity and bm25 dataframes
# create a new dataframe to store the concatenated data

merged_df = pd.merge(cosine_similarity_df, bm25_df, on=['model', 'temp', 'prompt'])



# filter the table to only include temp = 0.8
merged_df = merged_df[merged_df['temp'] == 0.8]

#average over all prompts
printable_df = pd.DataFrame(columns=['model', 'temp', 'cosine_similarity_avg', 'bm25_avg_score', 'mean_perplexity'])
perplexity = [13.13, 13.27]

for model in ['mb', 'ft']:
    perp = perplexity.pop()
    temp_df = merged_df[merged_df['model'] == model]
    printable_df.loc[len(printable_df)] = {'model': model, 'temp': 0.8, 'cosine_similarity_avg': temp_df['cosine_similarity_avg'].mean(), 'bm25_avg_score': temp_df['bm25_avg_score'].mean(), 'mean_perplexity': perp}



# create publication quality table from printable_df
printable_df = printable_df.round(3)

fig, ax = plt.subplots(figsize=(12, 4)) # set size frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
ax.set_frame_on(False)  # no visible frame
tabla = table(ax, printable_df, loc='center', cellLoc = 'center')
tabla.auto_set_font_size(False) # Activate set fontsize manually
tabla.set_fontsize(10) # if ++fontsize is necessary ++colWidths
tabla.scale(1.2, 1.2) # Table size
plt.savefig('./data/metrics/table.png')