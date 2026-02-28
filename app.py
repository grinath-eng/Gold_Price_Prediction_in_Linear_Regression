import gradio as gr

def calculate_gold_rate(usd_lkr):
    scaled_input = scaler.transform(np.array(usd_lkr).reshape(1, -1))
    prediction = regressor.predict(scaled_input)[0][0]
    return round(float(prediction), 2)


demo = gr.Interface(
    fn=calculate_gold_rate,
    inputs=gr.Number(label="USD Price Today"),
    outputs=gr.Number(label="Predicted Gold Price (LKR per gram)")
)

demo.launch()
