# Inverse Reinforcement Learning for LLMs

## Training env

```
conda create --n IRLforLLM python==3.11.7
conda activate IRLforLLM

git clone this repo
cd into the repo

cd src

pip install -r requirements.txt

install trlx: https://github.com/CarperAI/trlx
```

## Fine-Tune an LLM using RLHF
```(python)
python src/train_rlhf.py
```
## Implement Max-Margin IRL to extract the reward model from the RLHF'd LLM
```(python)
python irl.py
```