import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import joblib


## How to run this app:
## python -m streamlit run Football_Prediction.py ## This one work better
## streamlit run Football_Prediction.py


'''
    Commit 1 - Initial Commit - 27 January 2026
    
    Commit 2 - Did the Overview, EDA, Dataset Definitions and a mock design for the predictions - 28 January 2026

'''

icon = Image.open("Images/Icon3.png")

st.set_page_config(
    page_title="Football Player Value Prediction",
    layout="centered",
    page_icon=icon
)



@st.cache_data
def load_data():
    df = pd.read_csv("Experiment5_2_Dataset.csv")
    return df

df = load_data()



@st.cache_resource
def load_model_and_columns():
    model = joblib.load("BEST_FOOTBALL_MODEL_INTHEWORLD.joblib")  # change name if different
    feature_columns = joblib.load("BEST_FOOTBALL_MODEL_INTHEWORLD_COLUMNS.joblib")
    return model, feature_columns

model, feature_columns = load_model_and_columns()



## ======================= DESIGN START HERE =========================================== ##

st.title("Football Value Predictor 🤑⚽")
st.caption("MLDP Project - Football Transfer Valuation using Machine Learning")

st.image("Images/SEWYYY.png", caption="SEWYYYYYY")


st.divider()

st.subheader("Introduction")

st.markdown("""
            
            Ever wanted to be a football player but **failed** because you live in Singapore?   
            Ever wonder how much you would cost, but never had the chance to find out because your max market value is **€0**? Do not fret, this Streamlit website got you, bro. 😎

            This website is part of my **Machine Learning for Developers (MLDP)** project.  
            The goal is to predict and **find out how much you are worth** based on real football player market data.  
            I tuned a **Gradient Boosting Regressor** model trained on **FIFA-style player data**.

            The model doesn't just look at raw stats - it considers **technical abilities**, **tactical role**, **continent**, and more. Want to really find out how much you're worth if you were a football player?  
            **Come try out my website today!** 💸⚽

            Check out the dataset **[here](https://www.kaggle.com/datasets/maso0dahmed/football-players-data)**!
""")

st.divider()

st.subheader("Overview")
st.markdown("""
- [Data Dictionary 📚](#data-dict)
- [Exploratory Data Analysis 📊](#eda)
- [Try the Football Value Predictor 🤑💵💰💸](#predictor)
- [Results time BABY ⚽⚽⚽](#results)
""")

st.divider()

st.markdown('<a id="data-dict"></a>', unsafe_allow_html=True)
column_descriptions = {
    "age": "Player's age in years.",
    "height_cm": "Player height in centimetres.",
    "weight_kgs": "Player weight in kilograms.",
    "overall_rating": "Current overall ability rating (0-99).",
    "potential": "Ceiling rating the player can reach in their career (0-99).",
    "value_euro": "Current estimated market value in Euros (target variable).",
    "wage_euro": "Weekly wage paid to the player in Euros.",
    "preferred_foot": "Player's dominant foot: Left or Right.",
    "international_reputation(1-5)": "Global reputation score (1-5).",
    "weak_foot(1-5)": "How good the weaker foot is (1-5).",
    "skill_moves(1-5)": "Skill moves rating (1-5).",
    "primary_position": "Main position on the pitch (e.g. GK, CB, CM, ST, LW, RW).",
    "body_type_clean": "Simplified body type (Lean / Normal / Stocky / Other).",
    "continent": "Continent grouping based on nationality (e.g. Europe, South America).",
    "pace_index": "Average of pace-related stats (acceleration, sprint speed).",
    "shooting_index": "Average of shooting-related stats (finishing, shot power, long shots, etc.).",
    "passing_index": "Average of passing-related stats (short/long passing, vision, crossing, etc.).",
    "dribbling_index": "Average of dribbling-related stats (dribbling, ball control, agility, etc.).",
    "defending_index": "Average of defending-related stats (tackling, interceptions, heading, etc.).",
    "physical_index": "Average of physical stats (strength, stamina, jumping, aggression).",
    "growth": "Max(potential - overall_rating, 0). How much room the player still has to improve.",
    "position_group": "Broad tactical group: Goalkeeper, Defender, Midfielder, or Attacker."
}

dict_df = pd.DataFrame(
    [{"Column": col, "Description": desc} for col, desc in column_descriptions.items()]
)


st.subheader("Data Dictionary 📚")
st.dataframe(
    dict_df,
    hide_index=True

)

st.divider()

st.markdown('<a id="eda"></a>', unsafe_allow_html=True)
st.subheader("Data Visualisaion and EDA time baby 📊")

## ====================================================================== ##

st.markdown("#### Market Value by Primary Position 💰⚽")

if {"primary_position", "value_euro"}.issubset(df.columns):
    # Group by primary_position and take median value
    value_by_pos = (
        df.groupby("primary_position", as_index=False)["value_euro"]
          .median()
          .rename(columns={"value_euro": "median_value_euro"})
    )

    # Convert to millions for nicer scale
    value_by_pos["median_value_million"] = value_by_pos["median_value_euro"] / 1_000_000

    # Sort for nicer ordering in the bar chart
    value_by_pos = value_by_pos.sort_values("median_value_million", ascending=False)

    fig = px.bar(
        value_by_pos,
        x="primary_position",
        y="median_value_million",
        title="Median Market Value by Primary Position",
        labels={
            "primary_position": "Primary Position",
            "median_value_million": "Median Market Value (million €)"
        }
    )

    # Order categories by total descending (just like your boxplot example)
    fig.update_layout(xaxis={"categoryorder": "total descending"})

    st.plotly_chart(fig, use_container_width=True)


    st.markdown(f"""
In the dataset the highest value player position is the **Left Winger Role(LW)**. Having the median
of the value euro being about **1.1 million euros**. If you want to be worth as much, play as a 
Left Winger EHHEHEH, **Career Mode in real life!**



""")
else:
    st.warning("Columns `primary_position` and/or `value_euro` are missing from the dataset.")


## ==================================================================================================== ##


st.markdown("#### Market Value by Age ⏳💸")

if {"age", "value_euro"}.issubset(df.columns):
    # Group by age and take median value
    age_value = (
        df.groupby("age", as_index=False)["value_euro"]
          .median()
          .rename(columns={"value_euro": "median_value_euro"})
    )

    # Convert to millions for nicer scale
    age_value["median_value_million"] = age_value["median_value_euro"] / 1_000_000

    # Sort by age just in case
    age_value = age_value.sort_values("age")

    # Line chart
    fig_age = px.line(
        age_value,
        x="age",
        y="median_value_million",
        title="Median Market Value by Age",
        labels={
            "age": "Age",
            "median_value_million": "Median Market Value (million €)"
        }
    )

    st.plotly_chart(fig_age, use_container_width=True)

    st.markdown(f"""
From this dataset, players around **age 31** are peaking the hardest,
with a median value of about **€1.20 million.**

So if you are not in that age range, brother im sorry man, **no more Money...** 😕📉
                
But if you are coming of that age, congrats **my brother from another mother** 🤑🤱
""")
else:
    st.warning("Columns `age` and/or `value_euro` are missing from the dataset.")

## ============================================================================================= ##

st.markdown("#### Overall Rating vs Market Value 🌟💰")

if {"overall_rating", "value_euro"}.issubset(df.columns):
    # Copy and scale value for nicer axis
    df_scatter = df.copy()
    df_scatter["value_million"] = df_scatter["value_euro"] / 1_000_000

    # Scatter plot
    fig_scatter = px.scatter(
        df_scatter,
        x="overall_rating",
        y="value_million",
        title="Overall Rating vs Market Value",
        labels={
            "overall_rating": "Overall Rating (0-99)",
            "value_million": "Market Value (million €)"
        },
        opacity=0.35
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation for extra sauce
    corr = df_scatter[["overall_rating", "value_million"]].corr().iloc[0, 1]

    st.markdown(f"""
This chart is basically **FIFA logic in real life**:

- Higher overall rating → Usually **more money** 💸  
- Lower overall rating → **Bench Warmer** 💺 

The correlation between overall and value here is about **{corr:.2f}**,  
So you better get you **overall rating** up my guy, the higher it is the better your chances
of being valued higher than **1 Euros.**
""")
else:
    st.warning("Columns `overall_rating` and/or `value_euro` are missing from the dataset.")

st.divider()

st.markdown('<a id="predictor"></a>', unsafe_allow_html=True)
st.subheader("Prediction time my guy 🤑💵💰💸")

st.markdown("""
Fill in your **player stats** below and let the model tell you  
Lets see how much you are **worth my brother/sister.**
""")

# Use columns to make the form less tall
col1, col2, col3 = st.columns(3)

# ========== BASIC INFO ==========
with col1:
    age = st.number_input("Age", min_value=15, max_value=45, value=24)
    height_cm = st.number_input("Height (cm)", min_value=150, max_value=210, value=180)
    weight_kgs = st.number_input("Weight (kg)", min_value=50, max_value=110, value=75)

with col2:
    overall_rating = st.slider("Overall Rating (0-99)", min_value=40, max_value=99, value=80)
    potential = st.slider("Potential (0-99)", min_value=40, max_value=99, value=85)
    wage_euro = st.number_input("Weekly Wage (€)", min_value=0, max_value=1_000_000, value=50_000, step=5_000)

with col3:
    preferred_foot = st.selectbox("Preferred Foot", ["Right", "Left"])
    primary_position = st.selectbox(
        "Primary Position",
        ["GK", "CB", "LB", "RB", "LWB", "RWB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "ST", "CF"]
    )
    position_group = st.selectbox(
        "Position Group",
        ["Goalkeeper", "Defender", "Midfielder", "Attacker"]
    )


st.divider()


st.markdown("### Technical / Style Vibes 🎯")

col4, col5, col6 = st.columns(3)

with col4:
    international_rep = st.slider("International Reputation (1-5)", 1, 5, 1)
    weak_foot = st.slider("Weak Foot (1-5)", 1, 5, 3)
    skill_moves = st.slider("Skill Moves (1-5)", 1, 5, 3)

with col5:
    body_type_clean = st.selectbox("Body Type", ["Lean", "Normal", "Stocky", "Other"])
    continent = st.selectbox(
        "Continent",
        ["Europe", "South America", "North America", "Africa", "Asia", "Oceania"]
    )

with col6:
    pace_index = st.slider("Pace Index (0-99)", 0, 99, 80)
    shooting_index = st.slider("Shooting Index (0-99)", 0, 99, 75)
    passing_index = st.slider("Passing Index (0-99)", 0, 99, 75)

col7, col8, col9 = st.columns(3)

with col7:
    dribbling_index = st.slider("Dribbling Index (0-99)", 0, 99, 78)

with col8:
    defending_index = st.slider("Defending Index (0-99)", 0, 99, 60)

with col9:
    physical_index = st.slider("Physical Index (0-99)", 0, 99, 80)

# Growth is derived, not manually input
growth = max(potential - overall_rating, 0)

# ===== Build feature dict based on your model's expected columns =====
raw_features = {
    "age": age,
    "height_cm": height_cm,
    "weight_kgs": weight_kgs,
    "overall_rating": overall_rating,
    "potential": potential,
    "wage_euro": wage_euro,
    "preferred_foot": preferred_foot,
    "international_reputation(1-5)": international_rep,
    "weak_foot(1-5)": weak_foot,
    "skill_moves(1-5)": skill_moves,
    "primary_position": primary_position,
    "body_type_clean": body_type_clean,
    "continent": continent,
    "pace_index": pace_index,
    "shooting_index": shooting_index,
    "passing_index": passing_index,
    "dribbling_index": dribbling_index,
    "defending_index": defending_index,
    "physical_index": physical_index,
    "growth": growth,
    "position_group": position_group,
}

# Keep only columns your model actually expects, and order them correctly
input_df = pd.DataFrame([raw_features])
input_df = input_df[[col for col in feature_columns if col in input_df.columns]]

st.markdown("----")

if st.button("Predict my transfer value 🧾💸"):
    try:
        prediction = model.predict(input_df)[0]  # raw euros
        value_million = prediction / 1_000_000

        st.success(f"Your predicted market value is **€{prediction:,.0f}**  (~**€{value_million:,.2f} million**)")

        if value_million < 1:
            st.write("Ngl… that's **rotation player / free transfer** energy. But we move 🥲")
        elif value_million < 10:
            st.write("Solid player, respectable bag. **Europa League merchant** vibes. 😎")
        else:
            st.write("Okay superstar, relax. This is **Ballon d'Or conversation** money. 🏆🔥")
    except Exception as e:
        st.error("Something went wrong when making the prediction. Check the feature columns / model file.")
        st.text(str(e))



