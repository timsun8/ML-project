# PricePolice

PricePolice is a machine learning system for real estate price evaluation in Almaty, Kazakhstan.

The system predicts expected property prices using segmented CatBoost regression models and allows users to compare their own price with the predicted market value.

## Features

- Real estate price prediction
- Separate models for different property segments
- FastAPI backend
- Streamlit frontend
- Docker support
- Price comparison: overpriced, underpriced, or fair
- Warnings for unreliable predictions outside the training range

## Property Segments

The system uses separate models for:

- Apartments
- Houses
- Land plots
- Commercial properties

The final commercial model focuses on the main commercial property type: premises.

## Technologies

- Python
- Pandas
- NumPy
- CatBoost
- FastAPI
- Streamlit
- Docker

## Project Structure

```text
ML PROJ/
│
├── data/
│   ├── 2022Q1_Almaty_real_estate_transactions.xlsx
│   ├── 2022Q2_Almaty_real_estate_transactions.xlsx
│   └── ...
│
├── models/
│   ├── apartment_model.cbm
│   ├── commercial_model.cbm
│   ├── house_model.cbm
│   ├── land_model.cbm
│   ├── feature_columns.json
│   ├── rayons.json
│   └── streets.json
│
├── app.py
├── main.py
├── project.ipynb
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Machine Learning Pipeline

1. Data cleaning and preprocessing  
2. Feature engineering (`year`, `quarter`)  
3. Property-type segmentation  
4. Segment-specific filtering and outlier handling  
5. Location feature experiments (`kato`, `rayon`, `street`)  
6. Logarithmic transformation (`log_plosh`, `log(summa)`)  
7. Segment-specific CatBoost model training and hyperparameter tuning  
8. Overfitting analysis and model evaluation  
9. Final training of all segment models on full data  
10. FastAPI integration  
11. Streamlit frontend + Docker deployment

## Features Used

The final models use the following features:

- `form_sobs`
- `rayon`
- `street`
- `year`
- `quarter`
- `log_plosh`

Target variable:

```text
log(summa)
```

## Model Performance

| Segment | Test MAE |
|---|---:|
| Apartments | ~7M ₸ (MAE)|
| Commercial | ~15M ₸ (MAE)|
| Land plots | ~17M ₸ (MAE)|
| Houses | ~18M ₸ (MAE)|

Prediction quality depended strongly on:
- data quantity
- segment homogeneity
- feature quality

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn main:app --reload
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

Start Streamlit:

```bash
streamlit run app.py
```

## Docker

Build Docker image:

```bash
docker build -t pricepolice .
```

Run container:

```bash
docker run -p 8000:8000 -p 8501:8501 pricepolice
```

FastAPI:

```text
http://localhost:8000/docs
```

Streamlit:

```text
http://localhost:8501
```

## API Endpoint

### POST `/predict`

Example request:

```json
{
  "property_type": "Квартира",
  "form_sobs": "Индивидуальная",
  "rayon": "р-н Медеуский",
  "street": "пр. Достык",
  "year": 2024,
  "quarter": 2,
  "plosh_ob": 150,
  "user_price": 150000000
}
```

Example response:

```json
{
  "property_type": "Квартира",
  "predicted_price": 48366616.05,
  "predicted_price_mln": 48.37,
  "reliable": true,
  "user_price": 150000000,
  "difference": 101633383.95,
  "difference_mln": 101.63,
  "difference_percent": 210.13,
  "status": "overpriced",
  "message": "Your price is higher than expected by 210.13%."
}
```

## Future Improvements

- Add geospatial features
- Add coordinates and distance-based features
- Add building-level characteristics
- Improve handling of elite real estate
- Improve segmentation for commercial properties
- Add inflation or market trend adjustment