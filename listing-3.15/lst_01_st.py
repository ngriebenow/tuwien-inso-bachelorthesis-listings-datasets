from simpletransformers.classification import ClassificationModel
import pandas as pd

train_data = [["Baustellenfahrzeuge", 1], ["Medizinische Geraete", 0]]
train_df = pd.DataFrame(train_data)

eval_data = [["Raupenfahrzeug fuer Baustelleneinsatz", 1], ["Dialysegeraet fuer Medizinische Universitaet Wien", 0]]
eval_df = pd.DataFrame(eval_data)

model = ClassificationModel("bert", "bert-base-german-cased", use_cuda=False)

model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)

predictions, raw_outputs = model.predict(["Baustellenkran"])