# Train WildFire

Train WildFire dataset using pyro-vision training script

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zmdtXpognUSMXDWUCEtAEfAEqY-CZvgV#scrollTo=UJeHqS76_epf/]

## Install

Install pyro-vision first

```bash
pip install pyronear
```

Download Dataset

```bash
check import depuis gdrive
```

## Train

```
python train.py WildFire2/ --model rexnet1_0x --lr 1e-3 -b 16 --epochs 20 --opt radam --sched onecycle --device 0
```

