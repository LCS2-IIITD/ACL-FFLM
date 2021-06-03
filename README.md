# 🌳 Fingerprinting Fine-tuned Language Models in the wild <br>

This is the code and dataset for our ACL 2021 (Findings) Paper - <b> Fingerprinting Fine-tuned Language Models in the wild </b>.<br>

## Clone the repo

```bash
git clone https://github.com/LCS2-IIITD/ACL-FFLM.git
pip3 install -r requirements.txt 
```

## Dataset

The dataset includes both organic and synthetic text.
<ul>
  <li> <b> Synthetic </b> - 
    <p align = "justify">
      Collected from posts of <a href = "https://www.reddit.com/r/SubSimulatorGPT2/">r/SubSimulatorGPT2</a>. Each user on the  subreddit is  a  GPT2  small  (345  MB)  bot  that  is  fine-tuned   on   500k   posts   and   comments   from   a particular  subreddit  (e.g.,  r/askmen,  r/askreddit,r/askwomen). The  bots generate posts on r/SubSimulatorGPT2, starting off with the main post followed by comments (and replies) from other bots. The bots also interact with each other by using the synthetic text in the preceding comment/reply as their prompt. In total, the sub-reddit contains 401,214 comments posted between June  2019  and  January  2020  by  108  fine-tuned GPT2  LMs (or class).
     </p>
  <li> <b> Organic </b> - 
    <p align = "justify">
    Collected from comments of 108 subreddits the GPT2 bots have been fine-tuned upon. We randomly collected about 2000 comments between the dates of June 2019 - Jan 2020.
    </p>
</ul>

The complete dataset is available <a href = "https://drive.google.com/drive/folders/1r9129JJ3QTtF0r-aQ6fSXtSzbEb8RGHo?usp=sharing">here</a>.
Download the dataset as follows -

<ol>
  <li> Download the 2 folders organic and synthetic, containing the comments from individual classes.
  <li> Store them in the data folder in the following format.
</ol>

```bash
data
├── organic
├── synthetic
└── authors.csv
```

Note - <br>
For the below TL;DR run you also need to download ```dataset.json``` and ```dataset.pkl``` files which contain pre-processed data. <br>
Organize them in the ```dataset/synthetic``` folder as follows - <br>

```bash
dataset
├── organic
├── synthetic
  ├── splits (Folder already present)
    ├── 6 (Folder already present)
      └── 108_800_100_200_dataset.json (File already present)
  ├── dataset.json (to be added via drive link)
  └── dataset.pkl (to be added via drive link)
└── authors.csv (File already present)
```

```108_800_100_200_dataset.json``` is a custom dataset which contains the comment ID's, the labels and their separation into train, test and validation splits. <br>
Upon running the models, the comments for each split are fetched from the ```dataset.json``` using the comment ID's in the ```108_800_100_200_dataset.json file``` . <br>

## Running the code

#### TL;DR
You can skip the pre-processing and the Create Splits if you want to run the code on some custom datasets available in the ```dataset/synthetic....splits```  folder. Make sure to follow the instructions mentioned in the Note of the Dataset section for the setting up the dataset folders.

### Pre-process the dataset
<p align = "justify"> 
First, we pre-process the complete dataset using the data present in the folder create-splits. Select the type of data (organic/synthetic) you want to pre-process using the parameter synthetic in the file. By deafult the parameter is set for synthetic data i.e True. This would create a pre-processed dataset.json and dataset.pkl files in the dataset/[organic OR synthetic] folder.
</p>

### Create Train, Test and Validation Splits
<p align = "justify">
We create splits of train, test and validation data. The parameters such as min length of sentences (default 6), lowercase sentences, size of train (max and default 800/class), validation (max and default 100/class) and test (max and default 200/class),number of classes (max  and default 108) can be set internally in the create_splits.py in the splits folder under the commented PARAMETERS Section.
</p>
  
```bash
cd create-splits.py
python3 create_splits.py
```

This creates a folder in the folder ```dataset/synthetic/splits/[min_len_of_sentence/min_nf_tokens = 6]/```. 
The train, validation and test datasets are all stored in the same file with the filename   ```[#CLASSES]_[#TRAIN_SET_SIZE]_[#VAL_SET_SIZE]_[#TEST_SET_SIZE]_dataset.json``` like ```108_800_100_200_dataset.json```.
 

### Running the model
Now fix the same parameters in the seq_classification.py file.
To train and test the best model (Fine-tuned GPT2/ RoBERTa) - 

```bash
cd models/generate-embed/ft/
python3 seq_classification.py 
```

A ```results``` folder will be generated which will contain the results of each epoch.

Note - <br>
For the other models - pretrained and writeprints, first generate the embeddings using the files in the folders ```models/generate-embed/[pre-trained or writeprints]```.  The generated embeddings are stored in the ```results/generate-embed``` folder. Then, use the script in the ```models/classifiers/[pre-trained or writeprints]``` to train sklearn classifiers on generated embeddings. The final results will be in the ```results/classifiers/[pre-trained or writeprints]``` folder.


## 👪 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. For any detailed clarifications/issues, please email to nirav17072[at]iiitd[dot]ac[dot]in .

## ⚖️ License
[MIT](https://choosealicense.com/licenses/mit/)
