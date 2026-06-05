To train all models:
python -m src.train  

for seperate training:
python -m src.train --models custom_cnn   
python -m src.train --models vgg16
python -m src.train --models resnet50
python -m src.train --models efficientnet

Hyperparameter tuning (KerasTuner Hyperband — run on GPU/HPC):
python -m src.tune                              # tune all models, then retrain+save best
python -m src.tune --models custom_cnn          # tune one model
python -m src.tune --max-epochs 30 --factor 3   # control Hyperband budget
python -m src.tune --no-final-train             # search only (writes *_best_hp.json)
python -m src.tune --overwrite                  # start a fresh search (discard prior state)

  Outputs:
    models/{model}_best_hp.json      best hyperparameters found
    models/{model}_final.keras       best config retrained (used by evaluate/app)
    models/{model}_history.json      training history for plots
    tuning/{model}/                  raw KerasTuner trial state (resumable)

Or via the master pipeline (tune instead of plain train, then evaluate):
python run_pipeline.py --skip-download --tune
python run_pipeline.py --skip-download --tune --models custom_cnn --tune-max-epochs 30

To evaluate all models:
python -m src.evaluate

seperate evaluation:
python -m src.evaluate --models custom_cnn
python -m src.evaluate --models vgg16
python -m src.evaluate --models resnet50
python -m src.train --models efficientnet


UI:
streamlit run src/app.py
