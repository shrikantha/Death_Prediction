import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio as gr
from fastapi import FastAPI, Request, Response
import joblib
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# FastAPI object
app = FastAPI()

# Load your trained model

# YOUR CODE HERE
model = joblib.load("./xgboost-model.pkl")

# Function for prediction
def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):

    # YOUR CODE HERE...
    ip = np.array([age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]).reshape(1,-1)
    return model.predict(ip)

op = predict_death_event(7.00e+01, 0.00e+00, 1.61e+02, 0.00e+00, 2.50e+01, 0.00e+00,
       2.44e+05, 1.20e+00, 1.42e+02,1,1,1)                        
print(op)


# Gradio interface to generate UI link
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

# Define the interface
with gr.Blocks() as interface:
    with gr.Column():
        age = gr.Slider(minimum=0, maximum=100, step=1, label='Age')
        anaemia = gr.Radio([1,0], label='Anaemia')
        creatinine_phosphokinase = gr.Slider(minimum=0, maximum=1000, step=1, label='Creatinine Phosphokinase')
        diabetes = gr.Radio([1,0], label='Diabetes')
        ejection_fraction = gr.Slider(minimum=0, maximum=100, step=1, label='Ejection Fraction')
        high_blood_pressure = gr.Radio([1,0], label='High Blood Pressure')
        platelets = gr.Slider(minimum=0, maximum=1000, step=1, label='Platelets')
        serum_creatinine = gr.Slider(minimum=0, maximum=10, step=0.1, label='Serum Creatinine')
        serum_sodium = gr.Slider(minimum=0, maximum=150, step=1, label='Serum Sodium')
        sex = gr.Radio([1,0], label='Sex')
        smoking = gr.Radio([1,0], label='Smoking')
        time = gr.Slider(minimum=0, maximum=10, step=1, label='Time')

    submit_button = gr.Button('Submit')
    output = gr.Textbox()

    submit_button.click(fn=predict_death_event, inputs=[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time], outputs=output)


# Launch the interface
# interface.launch(share = True,debug=True)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface
# iface.launch(server_name="0.0.0.0", server_port = 8001)

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gr.mount_gradio_app(app, interface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
