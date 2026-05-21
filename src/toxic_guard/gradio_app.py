import gradio as gr

from toxic_guard.services.predictor import predict


def predict_text(text: str):

    result = predict(text)

    prediction = result["prediction"]

    probability = result.get("probability", {})

    return {
        "Toxic": probability.get("toxic", 0),
        "Non Toxic": probability.get("non_toxic", 0)
    }, prediction, result.get("clean_text", "")


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="ToxicGuard"
) as demo:

    gr.Markdown(
        """
        # ToxicGuard
        
        Phát hiện xem một bình luận có độc hại hay không.
        """
    )

    with gr.Row():

        with gr.Column(scale=2):

            text_input = gr.Textbox(
                label="Enter Comment",
                placeholder="Ví dụ: Đồ_ngốc quá!!",
                lines=6
            )

            predict_button = gr.Button(
                "Predict",
                variant="primary"
            )

        with gr.Column(scale=1):

            prediction_output = gr.Textbox(
                label="Prediction"
            )

            clean_text_output = gr.Textbox(
                label="Clean Text"
            )

    probability_output = gr.Label(
        label="Prediction Probability"
    )

    predict_button.click(
        fn=predict_text,
        inputs=[text_input],
        outputs=[
            probability_output,
            prediction_output,
            clean_text_output
        ]
    )

    gr.Examples(
        examples=[
            ["Bạn thật tuyệt vời"],
            ["Cút đi đồ khốn"],
            ["Tôi rất thích sản phẩm này"]
        ],
        inputs=text_input
    )


if __name__ == "__main__":

    demo.launch()