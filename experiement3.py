from transformers import pipeline

# ── A. Summarization ──────────────────────────────────────────────────────────

# Load the summarization pipeline (using small model for faster download)
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    framework="pt"
)

# Input text
text = """
Artificial Intelligence is transforming many industries by enabling
machines to perform tasks that normally require human intelligence.
It is widely used in healthcare, education, manufacturing, finance,
transportation, and cybersecurity. AI systems can analyze large
amounts of data, identify patterns, make predictions, and support
intelligent decision-making. Generative AI is a branch of Artificial
Intelligence that can create new content such as text, images, audio,
video, and computer programs.
"""

# Generate the summary
result = summarizer(
    text,
    max_length=60,
    min_length=20,
    do_sample=False
)

# Display the result
print("Summary:")
print(result[0]["summary_text"])

# ── B. Question Answering ─────────────────────────────────────────────────────

# Load the question-answering pipeline (force PyTorch backend)
question_answerer = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    framework="pt"
)

# Input context
context = """
Generative Artificial Intelligence is a type of Artificial Intelligence
that can create new content such as text, images, audio, video, and
computer programs. Large Language Models are commonly used for text
generation, summarization, translation, and question answering.
"""

# Input question
question = "What type of content can Generative AI create?"

# Predict the answer
result = question_answerer(
    question=question,
    context=context
)

# Display the result
print("\nAnswer:", result["answer"])
print("Confidence Score:", round(result["score"], 3))
