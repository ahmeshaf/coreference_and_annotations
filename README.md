# Coreference Resolution and Annotations
A single repository to do both Coreference Resolution and Machine-assisted Coreference Annotations

## Event and Entity Coreference Resolution

An experimental setup to run event and entity coreference resolution on LDC datasets.

### Prerequisites

#### Datasets
Contact: shah7567@colorado.edu
Annotations: 
Source: 

#### Python  requirements
```commandline 
pip install -r requirements.txt
```

### Running the pipeline
```commandline
python ./coreference/coref.py --ann path_to_annotation_folder --source path_to_source_folder -t tmp_folder -m evt
```

```commandline
usage: coref.py [-h] [--ann ANN] [--source SOURCE] [--tmp_folder TMP_FOLDER] [--men_type MEN_TYPE]

Run and evaluate cross-document coreference resolution on LDC Annotations

optional arguments:
  -h, --help            show this help message and exit
  --ann ANN, -a ANN     Path to the LDC Annotation Directory
  --source SOURCE, -s SOURCE
                        Path to the LDC Source Directory
  --tmp_folder TMP_FOLDER, -t TMP_FOLDER
                        Path to a working directory
  --men_type MEN_TYPE, -m MEN_TYPE
                        Mention type for coreference. Either evt or ent
```
### Running the pipeline with BERT Embeddings:
```commandline

python ./coref_bert.py --ann path_to_annotation_folder --source path_to_source_folder -t tmp_folder  --model Transformer model type used for coreference task --men_type Mention type for coreference --bert True


Run and evaluate cross-document coreference resolution on LDC Annotations with BERT embeddings 

optional arguments:
  -h, --help            show this help message and exit
  --ann ANN, -a ANN     Path to the LDC Annotation Directory
  --source SOURCE, -s SOURCE
                        Path to the LDC Source Directory
  --tmp_folder TMP_FOLDER, -t TMP_FOLDER
                        Path to a working directory
                        
  --model MODEL         
                        Transformer model type used for coreference task, currently bert_base, bert_large and bert_coref as options
  --men_type MEN_TYPE, -m MEN_TYPE
                        Mention type for coreference. Either evt or ent
                        
  --bert BERT_FLAG, -bert BERT_FLAG
                        Whether to use transformer embedding or lemma for coref. 'store_true' is set so ignore this argument for lemma coref
                                              
                        
                        
```



