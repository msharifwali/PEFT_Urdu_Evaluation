#Imports 📢
from datasets import Audio, load_metric, load_dataset, DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import time
from tqdm import tqdm
import sys
from jiwer import cer as jiwer_cer
import numpy as np
from Dataset_Preprocessing import preprocess_dataset
import evaluate
from typing import Any, Dict, List, Union
from dataclasses import dataclass
from torch.utils.data import DataLoader
import gc
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
#========================================================================
torch.cuda.is_available()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available() :
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3," #it was prviously the 0 for one GPU 
  print('yes')
  print(f"No of GPUs Available:  {torch.cuda.device_count()}")
#str(torchaudio.get_audio_backend())
#print(torch.__version__)
#print(torchaudio.__version__)

print("====================== 🧠 Load the PreTrained Model 🧠 =============================")
#model_name_or_path = "openai/whisper-tiny"

# model_name_or_path = "/data3/sharif/Datasets/openai_whisper_base"   
model_name_or_path = "/data3/sharif/Datasets/openai_whisper_tiny" #whisper_Tiny
# model_name_or_path = "/data3/sharif/Datasets/openaiwhisper-small"   #whisper_small
# model_name_or_path = "/data3/sharif/Datasets/openai_whisper-medium"   #whisper_medium
# model_name_or_path = "/data3/sharif/Datasets/openai_whisper-large-v3"   #whisper_large

print("Load the PreTrained Model:: Whisper ", model_name_or_path)

language = "Urdu"
language_abbr = "ur"
task = "transcribe"

#dataset_name = "mozilla-foundation/common_voice_13_0"
dataset_name = "/data3/sharif/Datasets/cv-corpus-9.0-2022-04-27/ur/" #DS CV-09
# dataset_name = "/data3/sharif/Datasets/cv-corpus-13.0-2023-03-09-ur/cv-corpus-13.0-2023-03-09/ur/" #DS CV13
# dataset_name ="/data3/sharif/Datasets/cv-corpus-19.0-2024-09-13/ur/"

eval_batch_size= 16 #44

#Model=================================================================== 
#==============**** tiny  base  small  medium  large ****================
#========================================================================
#path ="xyz/whisper-base-urdu"

path ="./whisper-urdu_V09/checkpoint-6000/" # its actually pointing the whisper pretrained model
# path = "/data3/sharif/ur_ir_wspr/Output_Train_Tiny_CV09/checkpoint-6000"  # its actually pointing the whisper pretrained model
# path ="/data3/sharif/Datasets/openai_whisper_base" # its actually pointing the whisper pretrained model

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
model = WhisperForConditionalGeneration.from_pretrained(path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

print("====================== 🧠 Model 🧠 =============================")
print(path)
print("Name Model:", model_name_or_path)
def print_size_of_model(model):
        path = "temp.p"
        torch.save(model.state_dict(), path)
        size = os.path.getsize(path)/1e6
        print(f"Size Model (MB): {size}")
        os.remove(path)
        return size

print_size_of_model(model)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

num_params = sum(param.numel() for param in model.parameters())   
print(f"Param Model: {num_params}")

num_params_LR = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f"Param lernebel: {num_params_LR}")

print("===================== 🗂️--- DataSet --- 🗂️ ============================")
common_voice = DatasetDict()
# common_voice= load_dataset("csv", data_files={"test": "/data3/sharif/Datasets/cv-corpus-13.0-2023-03-09-ur/cv-corpus-13.0-2023-03-09/ur/test.tsv"}, delimiter="\t")["test"]

print(" DATASET ::", dataset_name)
# common_voice["test"] = load_dataset("csv", data_files={"test": "/data3/sharif/Datasets/cv-corpus-13.0-2023-03-09-ur/cv-corpus-13.0-2023-03-09/ur/test.tsv"}, delimiter="\t")["test"]
# common_voice["test"] = load_dataset("csv", data_files={"test": "/data3/sharif/Datasets/cv-corpus-19.0-2024-09-13/ur/test.tsv"}, delimiter="\t")["test"]
common_voice["test"] = load_dataset("csv", data_files={"test": "/data3/sharif/Datasets/cv-corpus-9.0-2022-04-27/ur/test.tsv"}, delimiter="\t")["test"]

#added to check: SWK
common_voice["test"] = common_voice["test"].remove_columns(['client_id','up_votes', 'down_votes', 'age', 'gender', 'accents', 'locale', 'segment'])
common_voice["test"] = common_voice["test"].add_column("audio", [ dataset_name + '/clips/'  + pah for pah in common_voice["test"]['path']])


print(f"Test -set Len:  {len(common_voice['test'])}")
print("Before preprocessing:",common_voice['test']['sentence'][120]) 
print("Before preprocessing:",common_voice['test']['sentence'][200]) 
print("Before preprocessing:",common_voice['test']['sentence'][250]) 

print(f"Test-set Len:  {len(common_voice['test'])}")
#print(common_voice['sentence'][0]) 
# Pass the dataset to preprocessing functions
# common_voice = preprocess_dataset(common_voice)     Print("preprocessing Effect to see") 
#print(common_voice['sentence'][0]) 

#added:
print("Before preprocessing:",common_voice['test']['sentence'][120]) 
print("Before preprocessing:",common_voice['test']['sentence'][200]) 
print("Before preprocessing:",common_voice['test']['sentence'][250]) 

print("==================== 🎵 Pre-Audio 🎵 ===========================")

#input_str = common_voice['sentence'][1]
input_str = common_voice["test"][0]["sentence"] 
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

common_voice['test'] = common_voice['test'].cast_column("audio", Audio(sampling_rate=16000))
print(common_voice["test"][0])

print("===================== feature_extractor =========================")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice['test'] = common_voice['test'].map(prepare_dataset, num_proc=1)
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

print("================= Evaluation and Inference ======================")
#print(common_voice['test'][:4])
metric = evaluate.load("wer")
#wer_metric = load_metric("wer")  #added SWK
normalizer = BasicTextNormalizer()  # HM
eval_dataloader = DataLoader(
     common_voice['test'], 
     batch_size=eval_batch_size, 
     collate_fn=data_collator, 
     shuffle=False)  # HrbK
model.to(device)
model.eval()

num_params_LR_2 = sum(param.numel() for param in model.parameters() if param.requires_grad)   # HrbK
print(f"param learnable_2: {num_params_LR_2}")

for parameter in model.parameters():  # HrbK
    parameter.requires_grad = True #SKW

num_params_LR_3 = sum(param.numel() for param in model.parameters() if param.requires_grad)   # HrbK
print(f"param learnable_3: {num_params_LR_3}")    

list_pred = []
list_label = []
normalized_predictions = [] # HM
normalized_references = [] # HM

for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast(enabled=False):  # HrbK
        
       # with torch.no_grad(): # HrbK
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            list_pred.extend(decoded_preds)
            list_label.extend(decoded_labels)
            normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])    # HM
            normalized_references.extend([normalizer(label).strip() for label in decoded_labels])  # HM
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )

    del generated_tokens, labels, batch
    gc.collect()
wer = 100 * metric.compute()
normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
normalized_cer =100 *(jiwer_cer(normalized_references, normalized_predictions)) 
cer= 100 *( jiwer_cer(list_label, list_pred))
print("--------------------Results WER, CER--------------------") 
print("Model Name:", model_name_or_path)
print("Dataset Name:", dataset_name)
print(f"{wer=}")
print(f"{cer=}")
print(f"{normalized_wer=}")
print(f"{normalized_cer=}")
wer_rounded =round(wer,3)
cer_rounded =round(cer,3)
norm_wer_rounded =round(wer,3)
norm_cer_rounded =round(cer,3)

print(f"wer= {wer_rounded}")
print(f"cer={cer_rounded}")
print(f"normalized_wer={norm_wer_rounded}")
print(f"normalized_cer={norm_cer_rounded}")
# for pred,label in zip(normalized_predictions, normalized_references):
#   print(f"{label=}")
#   print(f"{pred=}")
#   print('---')

num_samples_to_display = 7  # Change this number as needed

print("\nSample Predictions and References:\n")

for i in range(min(num_samples_to_display, len(list_pred))):
    print(f"Reference : {list_label[i]}")
    print(f"Prediction: {list_pred[i]}")
    print("-" * 70)
