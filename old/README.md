# token-flow

Installing dev depenedncies

```
pip install requirements.txt
python -m spacy download en_core_web_sm
```


## Usage


Computing the network
```
python src/tokenflow.py digest ./text-definitions/dictionary
```

Running little prince
```
python src/tokenflow.py dev ./text-definitions/stories/little-prince-small.txt
```