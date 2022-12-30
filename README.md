
# Description
This is a application of flower identification.

The algorithm is developed based on CCT[https://arxiv.org/abs/2104.05704]. 

And the accuracy of it can reach 99.76%, which is state-of-art performance. 

# How to use

## Requirement
```text
Django==3.2.7
Pillow==9.3.0
timm==0.4.12
torch==1.9.0+cu111
torchvision==0.10.0+cu111
```

## Install
```bash
pip install -r requirements.txt
```

## Start
```bash
python manage.py runserver 0.0.0.0:8000
```

Open url:http://localhost:8000/


