
#Imports 📢

#This Script trains Urdu CV-19 finetuning the Pretrained Whisper PT model

#Sharif 
from datasets import Audio, load_dataset,DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer, AutoTokenizer
from transformers import WhisperProcessor
from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_metric, Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
import os
import sys
import torch.quantization
import csv
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import librosa
from datasets import load_from_disk
from transformers import Seq2SeqTrainer
from Dataset_Preprocessing import preprocess_dataset
from jiwer import cer as jiwer_cer
from evaluate import load

torch.cuda.empty_cache()
#========================================================================
torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#str(torchaudio.get_audio_backend())
#print(torch.__version__)
#print(torchaudio.__version__)

#Settings=============================================================
#model_name_or_path = "openai/whisper-base"
language = "Urdu"
language_abbr = "ur"
task = "transcribe"
#dataset_name = "mozilla-foundation/common_voice_18_0"

#Local Link Dataset of CV Urdu==============================================
# model_name_or_path = "/data3/sharif/Datasets/openai_whisper_tiny"   #whisper_large -SWK
# model_name_or_path = "/data3/sharif/Datasets/openai_whisper-large-v3"   #whisper_large

# model_name_or_path = "/data3/sharif/Datasets/openai_whisper-medium"   #whisper_large -SWK
model_name_or_path = "/data3/sharif/Datasets/openaiwhisper-small"   #whisper_small

#dataset_name = "/data3/sharif/Datasets/cv-corpus-13.0-2023-03-09-ur/cv-corpus-13.0-2023-03-09/ur/" #Corpus 13.0
# dataset_name = "/data3/sharif/Datasets/cv-corpus-9.0-2022-04-27/ur/"   #CV-Urdu_version_9
dataset_name = "/data3/sharif/Datasets/Urdu_Kathbath_Data/ur"  # Update with your actual dataset path

# dataset_name = "/data3/sharif/Datasets/cv-corpus-19.0-2024-09-13/ur"   #Corpus 19.0 -SWK
# [dataset: https://commonvoice.mozilla.org/en/datasets 5.8GB, 301 Hours, 454Voices, Type: mp3, Date: 18/9/2024]

Use_dataset_AV= False
freez_encoder_model= True

Numberـsteps= 600  #6000   #  #Change SWK
train_batch_size= 1
gradient_accumulation= 2
eval_batch_size= 1
learning_rate_= 6.15044e-05
warmup= 500
#Model=================================================================== 
#==============**** tiny  base  small  medium  large ****================
#========================================================================
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to("cuda")
model.generation_config.language = "ur"
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.max_length = 448

if freez_encoder_model:
    model.freeze_encoder()

print("======================  Model  =============================")

print("Name Model:", model_name_or_path)
def print_size_of_model(model):
        path = "temp.p"
        torch.save(model.state_dict(), path)
        size = os.path.getsize(path)/1e6
        print(f"Size Model (MB): {size}")
        os.remove(path)
        return size

print_size_of_model(model)

num_params = sum(param.numel() for param in model.parameters())   
print(f"Param Model: {num_params}")

num_params_LR = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f"Param lernebel: {num_params_LR}")

print("===================== 🗂️ DataSet 🗂️ ============================")
model_name_or_path = "/data3/sharif/Datasets/openaiwhisper-small"   #whisper_small
dataset_name = "/data3/sharif/Datasets/Urdu_Kathbath_Data/ur"  # Update with your actual dataset path

train_audio_dir = os.path.join(dataset_name, 'train')  # Path to train audio folder
test_audio_dir = os.path.join(dataset_name, 'test')  # Path to test audio folder

# Load the transcriptions (assuming .txt files contain the sentences)
train_transcriptions_path = os.path.join(dataset_name, 'train.txt')
test_transcriptions_path = os.path.join(dataset_name, 'test.txt')

# Read the transcriptions from the .txt files
def load_transcriptions(file_path):
    with open(file_path, 'r') as f:
        transcriptions = f.readlines()
    return [line.strip() for line in transcriptions]

train_transcriptions = load_transcriptions(train_transcriptions_path)
test_transcriptions = load_transcriptions(test_transcriptions_path)

# Load the audio files based on audio_id from the dataset
def get_audio_path(audio_id, dataset_type='train'):
   
    if dataset_type == 'train':
        audio_dir = train_audio_dir
    elif dataset_type== 'test':
        audio_dir = test_audio_dir
    else:
        raise ValueError(f"Unkonwn dataset type:{dataset_type}")
       
    # Construct the path to the .m4a file based on audio_id
    audio_path = os.path.join(audio_dir, f"{audio_id}.m4a")  # Adjust file extension if necessary
    if os.path.exists(audio_path):
        return audio_path
    else:
        raise FileNotFoundError(f"Audio file {audio_id}.m4a not found in {audio_dir}")
    
# Example: Constructing the dataset for train

# def create_dataset(transcriptions, dataset_type='train'):
#    # Determine the directory for audio files (train/test)
#     audio_dir = train_audio_dir if dataset_type == 'train' else test_audio_dir
   
#     # Get the list of audio files in the corresponding directory
#     audio_files = sorted(os.listdir(audio_dir))  # Sort to ensure matching order
   
#     # Check if the number of audio files matches the number of transcriptions
#     if len(audio_files) != len(transcriptions):
#         raise ValueError(f"The number of audio files ({len(audio_files)}) does not match the number of transcriptions ({len(transcriptions)})")
   
#     # Match audio files with transcriptions
#     matched_audio_paths = []
#     for idx, transcription in enumerate(transcriptions):
#         # Get the corresponding audio filename
#         audio_filename = audio_files[idx]  # Get the audio file from sorted list
#         audio_id = os.path.splitext(audio_filename)[0]  # Remove the file extension to get the ID (e.g., '1', 'audio_name')
       
#         # Construct the full path to the audio file
#         audio_path = os.path.join(audio_dir, audio_filename)
       
#         # Append the path to the list
#         matched_audio_paths.append(audio_path)
   
#     return matched_audio_paths

# Construct the train and test datasets
# train_transcriptions = load_transcriptions(train_transcriptions_path)
# test_transcriptions = load_transcriptions(test_transcriptions_path)

# train_dataset = create_dataset(train_transcriptions, dataset_type= 'train')
# test_dataset = create_dataset(test_transcriptions, dataset_type= 'test')

# dataset_dict = DatasetDict({

#     'train': train_dataset,
#     'test': test_dataset
# })

from datasets import Dataset, DatasetDict

def create_dataset(transcriptions, dataset_type='train'):
    """
    Creates a dataset by matching transcriptions with audio files.
    The audio filenames are taken directly from the train/test directories, and they match the transcription IDs.
   
    Args:
    - transcriptions (list): List of transcriptions loaded from the .txt file.
    - dataset_type (str): 'train' or 'test' to specify which dataset to use.
   
    Returns:
    - List of dictionaries containing audio file paths and transcriptions.
    """
    # Determine the directory for audio files (train/test)
    audio_dir = train_audio_dir if dataset_type == 'train' else test_audio_dir
   
    # Get the list of audio files in the corresponding directory
    audio_files = sorted(os.listdir(audio_dir))  # Sort to ensure matching order
   
    # Check if the number of audio files matches the number of transcriptions
    if len(audio_files) != len(transcriptions):
        raise ValueError(f"The number of audio files ({len(audio_files)}) does not match the number of transcriptions ({len(transcriptions)})")
   
    # Create the dataset with audio paths and transcriptions
    dataset = []
    for audio_filename, transcription in zip(audio_files, transcriptions):
        audio_path = os.path.join(audio_dir, audio_filename)  # Full path to audio file
        dataset.append({
            'audio': audio_path,  # Add audio path
            'transcription': transcription  # Add transcription
        })
   
    return dataset

# Load transcriptions for train and test sets
train_transcriptions = load_transcriptions(train_transcriptions_path)
test_transcriptions = load_transcriptions(test_transcriptions_path)

# Create train and test datasets
train_data = create_dataset(train_transcriptions, dataset_type='train')
test_data = create_dataset(test_transcriptions, dataset_type='test')

# Convert the datasets into Hugging Face Dataset format
train_dataset = Dataset.from_dict({
    'audio': [data['audio'] for data in train_data],
    'transcription': [data['transcription'] for data in train_data]
})

test_dataset = Dataset.from_dict({
    'audio': [data['audio'] for data in test_data],
    'transcription': [data['transcription'] for data in test_data]
})

# Combine them into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Now you can check the length of the datasets
print(f"Train-set Len:  {len(dataset_dict['train'])}")
print(f"Test-set Len:  {len(dataset_dict['test'])}")
# Checking the dataset structure
print(f"Train Dataset: {train_dataset}")
print(f"Test Dataset: {test_dataset}")
# Print to verify

print("===================== 🗂️ Data PreProcess 🗂️ ============================")

print("Before preprocessing:",train_dataset['transcription'][120]) 

# Pass the dataset to preprocessing functions

common_voice = preprocess_dataset(train_dataset)

print("After preprocessing :",train_dataset['transcription'][120]) 

print("====================  Pre-Audio  ===========================")

# input_str = common_voice[0]["transcription"] 
# labels = tokenizer(input_str).input_ids
# decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
# decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

# print(f"Input:                 {input_str}")
# print(f"Decoded w/ special:    {decoded_with_special}")
# print(f"Decoded w/out special: {decoded_str}")
# print(f"Are equal:             {input_str == decoded_str}")
print("==============**** tiny  base  small  medium  large ****================")

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to("cuda")
model.generation_config.language = "ur"
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.max_length = 448

if freez_encoder_model:
    model.freeze_encoder()

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Replace with your model's tokenizer

# Load the WhisperProcessor (tokenizer + feature extractor)
processor = WhisperProcessor.from_pretrained(model_name_or_path)

def preprocess_data(dataset):
    """
    Preprocess the dataset: tokenizes the transcriptions and processes the audio.
    """
    input_str = example["transcription"]  # Access the transcription text

    # Tokenize the transcription text
    labels = processor.tokenizer(input_str).input_ids
    decoded_with_special = processor.tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str = processor.tokenizer.decode(labels, skip_special_tokens=True)
       
    # Replace the `audio` column with the actual audio array using librosa
    audio_path = example["audio"] if isinstance(example["audio"], str) else example["audio"]["path"]
    audio, sampling_rate = librosa.load(audio_path, sr=16000)  # Load as 16kHz mono

    example["audio"] = audio
    example["sampling_rate"]= sampling_rate
    example["decoded_transcription"]= decoded_str

    return dataset

# Assuming train_dataset and test_dataset are already created
train_dataset = train_dataset.map(preprocess_data)
test_dataset = test_dataset.map(preprocess_data)

print("===================== feature_extraction =========================")
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)
#======================================================================== 
"""
You can activate the following code to save the features on the disk
"""
# #save feature in disk
# common_voice.save_to_disk("feature_train-filtered-Av-cv")
"""
You can activate the following code to read the features from the disk
"""
# #load datast
# common_voice = load_from_disk("feature_train-filtered-Av-cv")

#======================================================================== 
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print("#Debugging - SWK")
from evaluate import list_evaluation_modules
print(list_evaluation_modules)
#metric = evaluate.load("WER")   #Prevously utilized
#wer_metric = load_metric("wer")
wer_metric = load("wer")
print(wer_metric)
#========================================================================  
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

print("=====================  Training  ===========================")
print(f"Number of steps: {Numberـsteps}")
print(f"Batch size Train: {train_batch_size}")
print(f"Batch size Test:  {eval_batch_size}")
print(f"Gradient Accumulation: {gradient_accumulation}")
print(f"Learning Rate: {learning_rate_}")
print(f"Warmup: {warmup}")

#adopted from w2v2-xlsr - three lines
model.gradient_checkpointing_enable()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

training_args = Seq2SeqTrainingArguments(
    output_dir="./Output_Kathbath_Trained",  # REPO: its actually pointing the whisper small pretrained model
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps= gradient_accumulation,  # increase by 2x for every 2x decrease in batch size
    learning_rate=learning_rate_,
    warmup_steps= warmup,
    max_steps=Numberـsteps,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=eval_batch_size,
    predict_with_generate=True,
    generation_max_length=448,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
#model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train() 