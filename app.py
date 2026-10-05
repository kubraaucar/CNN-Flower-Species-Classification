import gradio as gr

from inference import predict_image


demo = gr.Interface(
    fn=predict_image,

    inputs=gr.Image(
        type="numpy",
        label="Çiçek Fotoğrafı Yükle"
    ),

    outputs=gr.Label(
        num_top_classes=3,
        label="Tahmin Sonuçları"
    ),

    title=" CNN Flower Classification",

    description=(
        "Bir çiçek fotoğrafı yükleyin. "
        "CNN modeli fotoğrafı analiz ederek "
        "çiçek türünü tahmin edecektir."
    )
)


if __name__ == "__main__":
    demo.launch()