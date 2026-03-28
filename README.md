# Personality Type Predictor

This project predicts an MBTI-style personality type after a user answers 10 short quiz questions. The quiz creates a persona summary, and that summary is classified by either a Decision Tree or KNN model trained on the Kaggle MBTI 500 dataset.

## Project Structure

- `app.py` - Streamlit application
- `train_model.py` - model training entry point
- `src/personality_predictor/` - quiz logic, config, charts, and ML helpers
- `assets/styles.css` - custom Streamlit styling
- `scripts/generate_reports.py` - DOCX report generator
- `models/` - saved joblib pipelines and metrics output
- `output/doc/` - generated project documents

## Setup

1. Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Place `MBTI 500.csv` in `data/`, or keep using the fallback path already configured in `src/personality_predictor/config.py`.

3. Train the models:

```powershell
python train_model.py
```

4. Generate the reports:

```powershell
python scripts/generate_reports.py
```

5. Run the app:

```powershell
streamlit run app.py
```

## Notes

- The app is designed for educational use and not as a clinical personality assessment.
- If no trained model is available yet, the app falls back to the direct quiz type until `train_model.py` is run.

