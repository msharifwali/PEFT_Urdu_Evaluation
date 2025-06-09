
#Imports 📢
from datasets import Audio, load_dataset,DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
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
if torch.cuda.is_available() :
  os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" #it was prviously the 0 for one GPU 
  print('yes')
  print(f"No of GPUs Available:  {torch.cuda.device_count()}")
#Settings🧰=============================================================
#model_name_or_path = "openai/whisper-base"
language = "Urdu"
language_abbr = "ur"
task = "transcribe"
#dataset_name = "mozilla-foundation/common_voice_18_0"

#Local Link Dataset of CV Urdu==============================================
# model_name_or_path = "/data3/sharif/Datasets/openai_whisper_tiny"   #whisper_Tiny
# model_name_or_path = "/data3/sharif/Datasets/openai_whisper_base"   #whisper_Base
# model_name_or_path = "/data3/sharif/Datasets/openai_whisper-medium"   #whisper_medium
# model_name_or_path = "/data3/sharif/Datasets/openai_whisper-large-v3"   #whisper_large
model_name_or_path = "/data3/sharif/Datasets/openaiwhisper-small"   #whisper_small

# dataset_name = "/data3/sharif/Datasets/cv-corpus-9.0-2022-04-27/ur/"   #CV-Urdu_version_9
dataset_name = "/data3/sharif/Datasets/cv-corpus-13.0-2023-03-09-ur/cv-corpus-13.0-2023-03-09/ur"   #CV-Urdu_13

Use_dataset_AV= False
freez_encoder_model= True

Numberـsteps= 6000
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
print("====================== 🧠 Model 🧠 =============================")

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
print(f"Param learnabel: {num_params_LR}")

print("===================== 🗂️ DataSet 🗂️ ============================")
common_voice = DatasetDict()

if Use_dataset_AV :

    ## just Arman_Av:
     common_voice["train"]= load_dataset("csv", data_files={"train": "/home/ubuntu/distil-whisper/training/final_code_data_set/train_AV_1403.csv"}, delimiter=",")["train"]
     common_voice["test"] = load_dataset("csv", data_files={"test": "/home/ubuntu/test_set_new.csv"}, delimiter=",")["test"]

else:
    #read train TSV and with out Arman_Av dataset
    # common_voice["train"]= load_dataset("csv", data_files={"train": "/data3/sharif/Datasets/cv-corpus-9.0-2022-04-27/ur/train.tsv"}, delimiter="\t")["train"]
    # common_voice["test"] = load_dataset("csv", data_files={"test": "/data3/sharif/Datasets/cv-corpus-9.0-2022-04-27/ur/test.tsv"}, delimiter="\t")["test"]
    common_voice["train"]= load_dataset("csv", data_files={"train": "/data3/sharif/Datasets/cv-corpus-13.0-2023-03-09-ur/cv-corpus-13.0-2023-03-09/ur/train.tsv"}, delimiter="\t")["train"]
    common_voice["test"] = load_dataset("csv", data_files={"test": "/data3/sharif/Datasets/cv-corpus-13.0-2023-03-09-ur/cv-corpus-13.0-2023-03-09/ur/test.tsv"}, delimiter="\t")["test"]
    common_voice["train"] = common_voice["train"].remove_columns(['client_id','up_votes', 'down_votes', 'age', 'gender', 'accents', 'variant', 'locale', 'segment'])


#Adding path of audio clips
common_voice["train"] = common_voice["train"].add_column("audio", [ dataset_name + '/clips/' + pah for pah in common_voice["train"]['path']])
common_voice["test"] = common_voice["test"].add_column("audio", [ dataset_name + '/clips/'  + pah for pah in common_voice["test"]['path']])

#print(common_voice)
print(f"Train-set Len:  {len(common_voice['train'])}")
print(f"Test -set Len:  {len(common_voice['test'])}")

print("Before preprocessing:",common_voice['train']['sentence'][120]) 

# Pass the dataset to preprocessing functions

common_voice = preprocess_dataset(common_voice)

print("After preprocessing :",common_voice['train']['sentence'][120]) 

print("==================== 🎵 Pre-Audio 🎵 ===========================")

input_str = common_voice["train"][0]["sentence"] 
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
print(common_voice["train"][0])

print("===================== feature_extractor =========================")
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
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

print("===================== 🎹 Training 🎹 ===========================")
print(f"Number of steps: {Numberـsteps}")
print(f"Batch size Train: {train_batch_size}")
print(f"Batch size Test:  {eval_batch_size}")
print(f"Gradient Accumulation: {gradient_accumulation}")
print(f"Learning Rate: {learning_rate_}")
print(f"Warmup: {warmup}")

#adopted from w2v2-xlsr - three lines
model.gradient_checkpointing_enable()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

training_args = Seq2SeqTrainingArguments(
    output_dir="./Output_Train_SMALL_13",  # change to a repo name of your choice
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

# Output:
# {'eval_loss': 0.7886870503425598, 'eval_wer': 34.64264962231261,
#   'eval_runtime': 2038.9295, 'eval_samples_per_second': 1.62, 
#   'eval_steps_per_second': 0.81, 'epoch': 5.81}