# DualEC

## Dependencies
```
pip install -r requirements.txt
```
## Dataset
1. [Kvasir-Capsule Dataset](https://osf.io/dv2ag/) and [Red Lesion Endoscopy Dataset](https://rdm.inesctec.pt/dataset/nis-2018-003)
2. [Endo4IE Dataset](https://data.mendeley.com/datasets/3j3tmghw33/1)
3. [Capsule endoscopy Exposure Correction (CEC) Dataset](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155161502_link_cuhk_edu_hk/EZuLCQk1SjRMr7L6pIpiG5kBwhcMGp1hB_g73lySKlVUjA?e=g84Zl8)

## Training
```
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/train.py -opt options/train/DualEC_CEC.yml
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/train.py -opt options/train/DualEC_KC.yml
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/train.py -opt options/train/DualEC_Endo4IE_UN.yml
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/train.py -opt options/train/DualEC_Endo4IE_Over.yml
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/train.py -opt options/train/DualEC_RLE.yml
```

## Testing
```
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/test.py -opt options/test/DualEC_CEC.yml
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/test.py -opt options/test/DualEC_KC.yml
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/test.py -opt options/test/DualEC_Endo4IE_UN.yml
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/test.py -opt options/test/DualEC_Endo4IE_Over.yml
PYTHONPATH="./:${PYTHONPATH}" python3 basicsr/test.py -opt options/test/DualEC_RLE.yml
```
