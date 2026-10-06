import gradio as gr

from inference import predict_image, MODEL_CONFIGS, DEFAULT_MODEL


demo = gr.Interface(
    fn=predict_image,

    inputs=[
        gr.Image(
            type="numpy",
            label="Çiçek Fotoğrafı Yükle"
        ),
        gr.Radio(
            choices=list(MODEL_CONFIGS),
            value=DEFAULT_MODEL,
            label="Model"
        ),
    ],

    outputs=gr.Label(
        num_top_classes=3,
        label="Tahmin Sonuçları"
    ),

    title=" CNN Flower Classification",

    description=(
        "Bir çiçek fotoğrafı yükleyin ve modeli seçin. "
        "Model fotoğrafı analiz ederek "
        "çiçek türünü tahmin edecektir."
    )
)


if __name__ == "__main__":
    demo.launch()
