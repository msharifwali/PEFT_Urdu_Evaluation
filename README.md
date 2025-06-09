# PEFT_Urdu_Evaluation

  This repository presents the PEFT LoRA based evaluation of Urdu ASR with multiple big datasets over multiple pretrained models

Machine:

  Distributed across four NVIDIA GeForce RTX 2080 Ti GPUs, each with 11 GB of memory. 
  
HyperParameters:

  configured the batch size as 1 for Medium and Large Models
  
  for smaller models, the batch size was kept as 3. 
  
  low-rank matrices of r=32, injected into the query and value projection layers of the attention mechanism. 
  
  implemented gradient accumulation,
  
  For larger datasets, learning rate of 1e-3 
  
  smaller datasets : 1e-5. Additionally, 
  
  500 warmup steps incorporated to ensure training stability.

Result Sample: 
  
  wer= 41.132
  cer=15.857
  normalized_wer=41.132
  normalized_cer=15.857

Sample Predictions and References:
  
  Model: /data3/sharif/Datasets/openaiwhisper-small
  Dataset : /data3/sharif/Datasets/Fluers_Urdu_Data

  Reference : اس نے مزید کہا کہ تاہم ان سے اس ذمہ داری کا نا پوچھا جائے جو ان کی ترقی کے مرحلے فرض اور صلاحییتوں سے آگے بڑھ گئے ہیں
  Prediction:  تحم اس نے مزید کہا کہ ان سے اس دیمداری کا نا پوچھا جائے جو ان کی ترقیقے مرحلے فرز اور سلحیتوں سے آگے بڑھ گئے ہیں
  ----------------------------------------------------------------------
  Reference : نتیجے کے طور پر کسی رکاوٹ پر قابو پانے کے لئے مل کر کام کرنے والی کسی تنظیم کا عمل صارف کی ضرورت کو پورا کرنے کے لئے ایک جدید اختراعی عمل کا باعث بن سکتا ہے
  Prediction:  نتیجے کا طور پر کسی رقاورت پر کابو پانے کے لئے ملکر کام کرنے والے کسی تنظیم کا عمل سارف کے ضرورت کو پرورا کرنے کے لئے ایک جدید ایک طرحی عمل کا بائس بن سکتا ہے
  ----------------------------------------------------------------------
  Reference : ان کے منظم دفاع بال ہینڈلنگ کی مہارت اور عمدہ ٹیم ورک نے انھیں پامردی سے کھڑے ہونے کا حوصلہ دیا اور یہ بات واضح ہوگئی کہ یہی ٹیم شکست دینے والی ہے
  Prediction:  ان کے امنازم دفعہ بال ہینٹلنگ کی محارت اور امدہ ٹیم ورک نے انہیں پامردی سے کھڑے ہونے کا حاصلہ دیا اور یہ بات واضح ہو گئی کہ یہی ٹیم شکرس تینے والی ہے
  ----------------------------------------------------------------------
  Reference : ان کے منظم دفاع بال ہینڈلنگ کی مہارت اور عمدہ ٹیم ورک نے انھیں پامردی سے کھڑے ہونے کا حوصلہ دیا اور یہ بات واضح ہوگئی کہ یہی ٹیم شکست دینے والی ہے
  Prediction:  ان کے منظم دفعہ بال ہینڈلنگ کی مہارت اور امدہ ٹیم ورک نے انہیں پامردی سے کھڑے ہونے کا حسلہ دیا اور یہ بات واضح ہو گئی کہ یہی ٹیم شکستے نے والی ہے
