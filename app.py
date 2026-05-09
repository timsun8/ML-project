import json
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="PricePolice",
    layout="centered"
)


with open("models/rayons.json", "r", encoding="utf-8") as f:
    RAYONS = json.load(f)

with open("models/streets.json", "r", encoding="utf-8") as f:
    STREETS = json.load(f)


st.markdown("""
<style>
.block-container {
    max-width: 850px;
    padding-top: 3rem;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 2rem;
    text-align: center;
}

.result-card {
    padding: 1.2rem;
    border-radius: 12px;
    background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
    margin-top: 1rem;
}

.price {
    font-size: 32px;
    font-weight: 700;
}

.small-text {
    color: #666;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">PricePolice</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Сервис оценки рыночной стоимости недвижимости</div>',
    unsafe_allow_html=True
)


property_type = st.selectbox(
    "Тип недвижимости",
    ["Квартира", "Дом", "Земельный участок", "Помещение"]
)

form_sobs = st.selectbox(
    "Форма собственности",
    ["Индивидуальная", "Общая совместная", "Общая долевая"]
)

rayon = st.selectbox(
    "Район",
    RAYONS
)

street = st.selectbox(
    "Улица",
    STREETS
)

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Год",
        min_value=2022,
        max_value=2026,
        value=2024,
        step=1
    )

with col2:
    quarter = st.selectbox(
        "Квартал",
        [1, 2, 3, 4]
    )

plosh_ob = st.number_input(
    "Площадь, м²",
    min_value=0.1,
    value=50.0,
    step=1.0
)

use_user_price = st.checkbox("Сравнить с моей ценой")

user_price = None
if use_user_price:
    user_price = st.number_input(
        "Ваша цена, ₸",
        min_value=1.0,
        value=30_000_000.0,
        step=1_000_000.0
    )


if st.button("Оценить стоимость"):
    payload = {
        "property_type": property_type,
        "form_sobs": form_sobs,
        "rayon": rayon,
        "street": street,
        "year": int(year),
        "quarter": int(quarter),
        "plosh_ob": float(plosh_ob)
    }

    if user_price is not None:
        payload["user_price"] = float(user_price)

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code != 200:
            st.error("API returned an error.")
            st.code(response.text)

        else:
            data = response.json()

            st.markdown("### Результат оценки")

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="small-text">Ожидаемая стоимость</div>
                    <div class="price">{data["predicted_price_mln"]} млн ₸</div>
                    <div class="small-text">Тип недвижимости: {data["property_type"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if "status" in data:
                st.markdown("### Сравнение с вашей ценой")

                if data["status"] == "fair":
                    st.success(data["message"])
                elif data["status"] == "overpriced":
                    st.error(data["message"])
                elif data["status"] == "underpriced":
                    st.info(data["message"])

                st.write(f"Разница: {data['difference_mln']} млн ₸")
                st.write(f"Отклонение: {data['difference_percent']}%")

            if not data.get("reliable", True):
                st.markdown("### Предупреждения")
                for warning in data.get("warnings", []):
                    st.warning(warning)

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI. Make sure the API server is running.")
        st.code("uvicorn main:app --reload")

    except requests.exceptions.Timeout:
        st.error("API request timed out.")

    except Exception as e:
        st.error("Unexpected error.")
        st.code(str(e))