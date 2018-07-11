# Home Credit Default Risk

## Setup

1- Clone current repo in desired parent folder:  
`git clone https://github.com/bielrv/Home-Credit-Default-Risk.git`

2- Create virtual environment `venv` using virtualenv:  
`virtualenv -p python3 venv`

3- Activate virtual environment:  
`source venv/bin/activate`

4- Install repo dependencies:  
`pip install -r requirements.txt`

5- Download competition dataset:   
`kaggle competitions download -c home-credit-default-risk`  
*(kaggle library already in requirements)*

6- (Optional) - If you are using a kernel you might need to run:  
`python3 -m pip install ipykernel`    
`python3 -m ipykernel install --user`  

## Notes

Use `pip freeze > requirements.txt` to refresh requirements
