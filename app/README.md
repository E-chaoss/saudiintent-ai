# Task 6: Prototype development and system testing

| | |
|---|---|
| **Owner** | ___ |
| **Status** | Not started in this repo |
| **Inputs** | The best model chosen by Task 5 |
| **Outputs** | Working web prototype, system testing results, live demo |

## What to build
- A simple web page: a box for the customer message, a button, and the predicted intent (with confidence).
- Connect it to the best model. For MARBERTv2: `from marbert.predict import IntentPredictor` (add `src/` to the import path).
- Test it with different messages and handle errors (empty message, model not loaded).
- Proposed hosting for the live demo: Hugging Face Spaces (Gradio), with the model on the Hugging Face Hub. To be confirmed.
- Add a `docs/DEMO.md` with the steps to run the demo for the final presentation.
