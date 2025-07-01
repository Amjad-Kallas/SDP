# Diabetes Feature Analysis with AI Assistant

This application provides interactive data analysis and visualization for diabetes datasets, powered by traditional ML/statistics and a local Llama-2 AI model for natural language queries.

---

## 1. Required Libraries

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- streamlit
- llama-cpp-python

---

## 2. Environment Setup

### Create and Activate a Virtual Environment

It is recommended to use a virtual environment to avoid dependency conflicts. You can create and activate one as follows:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
### Llama-2 Model (for AI Assistant)

The AI assistant uses a local Llama-2 model via `llama-cpp-python`.  
**A compatible GGUF model file** must be downloaded (`llama-2-7b-chat.Q4_K_M.gguf`) and placed in the project root.

**download link:**  
[llama-2-7b-chat.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)

**After downloading:**
The `model_path` in `naturalLanguage.py` should be updated to match the path of the downloaded file.

## 3. How to Run the Application

### Streamlit Web App

Start the app with:

```bash
streamlit run streamlit_app.py
```

- The app will open in your browser.
- Use the sidebar for traditional analysis.
- Use the main section for natural language queries (e.g., "plot decision boundary between Glucose and BMI").
- **The integrated AI assistant interprets natural language commands (for example 'plot mean BMI for OK class') using a local Llama-2 model.**

### Command Line

You may also run scripts like `main.py` for custom analysis ...

---

## 4. Format of Input Datasets

The main dataset is `diabetes.csv` (also supports `diabetes_short.csv` for testing).

**CSV columns:**

| Column Name                  | Description                        |
|------------------------------|------------------------------------|
| Pregnancies                  | Number of times pregnant           |
| Glucose                      | Plasma glucose concentration       |
| Blood Pressure                | Diastolic blood pressure (mm Hg)   |
| Skin Thickness                | Triceps skin fold thickness (mm)   |
| Insulin                      | 2-Hour serum insulin (mu U/ml)     |
| BMI                          | Body mass index (weight/height^2)  |
| Diabetes Pedigree Function (DPF)     | Diabetes pedigree function         |
| Age                          | Age (years)                        |
| Outcome                      | Class variable (OK, KO)    |

**Labeling context:**
- `Outcome = 1` means **OK** (diabetic)
- `Outcome = 0` means **KO** (non-diabetic)

This labeling is used throughout the project to distinguish between diabetic and non-diabetic cases.

**Example rows:**
```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
...
```

---

## 5. References

- Llama-2 model: [Meta AI](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 
- diabetes dataset: [dataset source](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download)

