import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import base64
import io
from PyPDF2 import PdfReader
import docx
import re
import pandas as pd

# App title and configuration
st.set_page_config(page_title="ExamPredictor AI", layout="wide")
st.title("📝 Exam Question Predictor")

# Model selection - focusing on Phi-1.5 for now
MODEL_OPTIONS = {
    "Phi-1.5 (500M)": "microsoft/phi-1_5",
    # Add other models later if needed
    # "TinyLlama (500M)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "Gemma (500M)": "google/gemma-2b-it",
}

# Cache model loading to prevent reloading
@st.cache_resource
def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Check if GPU is available and set appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model with appropriate device
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        model.eval()  # Set model to evaluation mode
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

# File processing functions
def extract_text_from_pdf(file):
    try:
        # Reset file pointer to beginning before reading
        file.seek(0)
        pdf_reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return ""

def extract_text_from_docx(file):
    try:
        # Reset file pointer to beginning before reading
        file.seek(0)
        doc = docx.Document(io.BytesIO(file.read()))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"DOCX extraction error: {str(e)}")
        return ""

def extract_text_from_file(file):
    try:
        if file.type == "application/pdf":
            return extract_text_from_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_text_from_docx(file)
        elif file.type == "text/plain":
            # Reset file pointer and read text
            file.seek(0)
            return file.getvalue().decode("utf-8")
        else:
            st.error(f"Unsupported file type: {file.type}")
            return ""
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return ""

# Sidebar for model selection and configuration
with st.sidebar:
    st.header("Model Settings")
    selected_model = st.selectbox("Select AI Model", list(MODEL_OPTIONS.keys()))
    model_key = MODEL_OPTIONS[selected_model]
    
    with st.expander("Advanced Settings"):
        temperature = st.slider("Creativity (Temperature)", 0.1, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Response Length", 256, 1024, 512)
    
    st.header("About")
    st.markdown("""
    This tool predicts exam questions by analyzing course content.
    
    **Features:**
    - Upload syllabus/notes or enter text
    - Generate questions for specific topics
    - View topic probabilities (mock)
    
    *Built with Streamlit and Phi-1.5*
    """)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Input Course Material")
    input_method = st.radio("Select input method:", ["Upload File", "Enter Text"])
    
    content = ""
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload syllabus, notes, or textbook", type=["pdf", "txt", "docx"])
        if uploaded_file:
            with st.spinner("Extracting text from document..."):
                content = extract_text_from_file(uploaded_file)
                if content:
                    st.success(f"Successfully extracted {len(content)} characters")
                    with st.expander("Preview extracted content"):
                        st.text(content[:500] + "..." if len(content) > 500 else content)
                else:
                    st.warning("No text extracted. Try another file.")
    else:
        content = st.text_area("Enter course material:", height=300, placeholder="e.g., Quadratic equations: ax^2 + bx + c = 0...")
    
    course_info = st.text_input("Course/Subject Name", value="Algebra")
    exam_type = st.selectbox("Exam Type", ["Final Exam", "Midterm", "Quiz", "Assessment"])
    difficulty = st.select_slider("Question Difficulty", options=["Basic", "Intermediate", "Advanced"])
    num_questions = st.slider("Number of questions to generate", 1, 10, 3)
    topic_options = ["General", "Quadratic Equations", "Linear Equations", "Geometry", "Trigonometry"]
    topic = st.selectbox("Focus Topic", topic_options)

with col2:
    st.header("Predicted Questions")
    
    # Fix the condition for generating questions
    generate_button = st.button("Generate Questions", type="primary")
    
    if generate_button:
        if not content and topic == "General":
            st.warning("Please provide course content or select a specific topic.")
        else:
            # Load model
            with st.spinner(f"Loading {selected_model}..."):
                tokenizer, model, error = load_model(model_key)
                
                if error:
                    st.error(f"Error loading model: {error}")
                else:
                    # Prepare generation
                    with st.spinner("Generating questions..."):
                        # Create prompt
                        prompt = f"""You are an expert professor in {course_info if course_info else 'this subject'}.

                        Create {num_questions} {difficulty} level questions for a {exam_type}.
                        {"Focus on " + topic + "." if topic != 'General' else 'Use the following course material.'}
                        
                        {"COURSE MATERIAL: " + content[:3000] if content else ""}
                        
                        FORMAT YOUR RESPONSE AS:
                        1. First question
                        2. Second question
                        ...and so on.
                        """
                        
                        # Tokenize and generate
                        try:
                            inputs = tokenizer(prompt, return_tensors="pt")
                            # Move inputs to the same device as the model
                            inputs = {k: v.to(model.device) for k, v in inputs.items()}
                            
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=0.95,
                                    do_sample=True
                                )
                            
                            # Extract response
                            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            # Find the start index of the first digit followed by a period
                            match = re.search(r'\d+\.', generated_text[len(prompt):])
                            if match:
                                start_idx = len(prompt) + match.start()
                                response = generated_text[start_idx:]
                            else:
                                response = generated_text[len(prompt):]
                            
                            # Display questions
                            st.markdown("### Generated Questions")
                            st.markdown(response)
                            
                            # Mock topic probabilities
                            topics = ["Quadratic Equations", "Linear Equations", "Geometry", "Trigonometry"]
                            mock_probs = [0.4, 0.3, 0.2, 0.1] if topic == "General" else [0.6 if t == topic else 0.2/(len(topics)-1) for t in topics]
                            st.write("### Predicted Topic Probabilities (Mock)")
                            results = pd.DataFrame({"Topic": topics, "Probability": mock_probs})
                            st.table(results.sort_values(by="Probability", ascending=False))
                            
                            # Download option
                            csv_content = "Question\n" + "\n".join([line.strip() for line in response.split("\n") if line.strip()])
                            b64 = base64.b64encode(csv_content.encode()).decode()
                            filename = f"{course_info if course_info else 'exam'}_questions.csv"
                            st.download_button(
                                label="Download Questions (CSV)",
                                data=csv_content,
                                file_name=filename,
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Error generating questions: {str(e)}")
                            st.error(f"Error details: {type(e).__name__}")
    else:
        st.info("Provide course material or select a topic, then click 'Generate Questions'.")

# Usage tips
with st.expander("Tips for best results"):
    st.markdown("""
    - **Upload detailed materials** for better context.
    - **Choose a specific topic** to focus question generation.
    - **Adjust difficulty** to match your exam needs.
    - **Set temperature >0.7** for creative questions.
    """)
