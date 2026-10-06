import gradio as gr

from inference import predict_image, MODEL_CONFIGS, DEFAULT_MODEL


demo = gr.Interface(
    fn=predict_image,

    inputs=[
        gr.Image(
            type="numpy",
            label="Upload a flower photo"
        ),
        gr.Radio(
            choices=list(MODEL_CONFIGS),
            value=DEFAULT_MODEL,
            label="Model"
        ),
    ],

    outputs=gr.Label(
        num_top_classes=3,
        label="Predictions"
    ),

    title="Flower Species Classification",

    description=(
        "Upload a flower photo and choose a model. "
        "The model predicts the species: daisy, dandelion, "
        "rose, sunflower or tulip."
    )
)


if __name__ == "__main__":
    demo.launch()
