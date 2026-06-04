To train all models:
python -m src.train  

for seperate training:
python -m src.train --models custom_cnn   
python -m src.train --models vgg16
python -m src.train --models resnet50
python -m src.train --models efficientnet

To evaluate all models:
python -m src.evaluate

seperate evaluation:
python -m src.evaluate --models custom_cnn
python -m src.evaluate --models vgg16
python -m src.evaluate --models resnet50
python -m src.train --models efficientnet


UI:
streamlit run src/app.py
