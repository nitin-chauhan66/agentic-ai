import time

import gradio as gr

from generation import generate_rag_response


def process_query_stream(query, search_type, model_type):
    """Process the query and stream the response more efficiently"""
    full_response = ""
    for chunk in generate_rag_response(query, search_type, 5, model_type, stream=True):
        full_response += chunk

        # Only yield every few characters to reduce UI updates
        if len(chunk) > 10 or chunk.endswith((".", "!", "?", "\n")):
            time.sleep(0.01)  # Small delay for smoother updates
            yield full_response

    # Ensure final text is always yielded
    yield full_response


def process_query_normal(query, search_type, model_type):
    """Process the query and return the complete response"""
    return generate_rag_response(query, search_type, 5, model_type, stream=False)


# Create Gradio interface
with gr.Blocks(title="LocalRAG Q&A System", theme="soft", css="""
    .custom-md-box {
        height: 60vh; 
        overflow-y: auto;
    }
    .box-border {
        border: 2px solid #4A90E2;
        padding: 16px;
        border-radius: 8px;
        background-color: #f9f9f9;
    }
""") as demo:
    gr.Markdown("# 📚 LocalRAG Q&A System")
    gr.Markdown(
        "Ask questions about the RAG paper and get answers using RAG technology!"
    )

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about Retrieval-Augmented Generation...",
                lines=4,
            )

            with gr.Row():
                search_type = gr.Radio(
                    ["keyword", "semantic", "hybrid"],
                    label="Search Method",
                    value="hybrid",
                    info="Choose how to retrieve relevant information",
                )

                model_type = gr.Radio(
                    ["gemini"],
                    label="AI Model",
                    value="gemini",
                    info="Select which model generates your answer",
                )

            stream_checkbox = gr.Checkbox(
                label="Stream Response",
                value=True,
                info="See the answer as it's being generated",
            )

            submit_btn = gr.Button("Generate Answer", variant="primary")

        with gr.Column(scale=2, elem_classes="box-border"):
            output = gr.Markdown(value="Answer", elem_classes="custom-md-box")

    # Handle form submission based on streaming preference
    def on_submit(query, search_type, model_type, stream):
        if not query.strip():
            return "Please enter a question."

        # Initial feedback to user
        yield (
            "Retrieving relevant information..."
            if stream
            else "Retrieving relevant information..."
        )

        if stream:
            yield from process_query_stream(query, search_type, model_type)
        else:
            return process_query_normal(query, search_type, model_type)

    submit_btn.click(
        on_submit,
        inputs=[query_input, search_type, model_type, stream_checkbox],
        outputs=output,
        show_progress="minimal",  # Add this for visual feedback
    )

# Launch the app
if __name__ == "__main__":
    demo.queue().launch()