# FA-Amy

**FA-Amy: A Deep Learning Framework for Amyloid Protein Prediction**

---

##  Introduction
**FA-Amy** is a deep learning-based framework designed to accurately predict amyloid proteins by integrating **global and local features**.  
Our model demonstrates strong robustness even under **imbalanced dataset conditions**, making it practical for real-world biological applications.

---

##  Features
- End-to-end deep learning framework
- Large learning model extracted features
- Incorporates **global + local  features**
- Robust performance on **imbalanced datasets**
- Easy-to-use training and evaluation scripts

---

##  Usage


FA-Amy relies on a large-scale pre-trained protein language model: ESM C. The model is implemented using Hugging Face's Transformers library and PyTorch. Please make sure to install the required dependencies beforehand.

ESM C: https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12

First, you need to download the model weights for ESM C  from the provided Hugging Face URLs. Please visit the above links to download the respective weight files.

Save the downloaded weight files in your working directory and make sure you know their exact paths.

Next, you will use the provided `generate_embeddings.py` script to generate embedding features with ESM C. In script, you need to modify the file paths according to your needs.Run the following commands:
```bash
cd src
python generate_embeddings.py
```
This will generate the corresponding embedding features files.

Next, you can proceed with model training and validation using the provided `train.py` script. Before running it, make sure you have prepared the training features and modified the file paths and other parameters according to your needs. Run the following command to start the model training and validation:
```bash
python train.py
```
After the script finishes running, it will generate the best model file for each fold in the five-fold cross-validation, as well as the final saved model file.

Finally, you can test the model on a independent dataset using the provided `test.py` script. Make sure you have prepared the test features and modified the file path of saved model according to your needs. Run the following command to test the model:
```bash
python test.py
```

##  Dataset

The benchmark dataset and the generalization dataset used in this study is included in the `Dataset` folder.

##  Concat
