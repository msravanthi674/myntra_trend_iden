import streamlit as st
import pandas as pd
import time
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

# Initialize pytrends
pytrends = TrendReq()

# Streamlit app
st.title("Google Trends Analysis for Fashion Trends")

# Import trend2.csv directly
trend_file = "trend2.csv"
try:
    # Attempt to read with utf-8 encoding
    df = pd.read_csv(trend_file, encoding='utf-8')
except UnicodeDecodeError:
    # If utf-8 fails, try with ISO-8859-1 encoding
    df = pd.read_csv(trend_file, encoding='ISO-8859-1')

# Display the dataframe
st.write("Loaded trend2.csv data:")
st.dataframe(df)

# Parameters
cutoff = st.slider("Number of top queries to process", min_value=1, max_value=200, value=50)
pause = st.slider("Pause time between API calls (seconds)", min_value=1, max_value=10, value=1)
timeframe = st.selectbox("Timeframe for Google Trends", options=["today 1-m", "today 3-m", "today 12-m"])
geo = 'IN'  # Set to India

# Limit the number of rows to process
df = df[:cutoff]

# Initialize the result dataframe
result_df = pd.DataFrame(columns=['Keyword', 'Trend', 'Frequency', 'Image'])

# Process each keyword and get the trend
for index, row in df.iterrows():
    keyword = row['product_name']
    image_url = row['image_url']  # Assuming you have an 'image_url' column with local paths
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [keyword]

    # Retry logic for TooManyRequestsError
    retries = 3
    success = False
    while retries > 0 and not success:
        try:
            pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo=geo, gprop='')
            df2 = pytrends.interest_over_time()
            success = True
        except TooManyRequestsError:
            retries -= 1
            time.sleep(pause * 5)  # Increase the pause for retries

    if not success:
        trend = 'N/A'
        frequency = 0
    else:
        try:
            trend1 = int((df2[keyword][-5] + df2[keyword][-4] + df2[keyword][-3]) / 3)
            trend2 = int((df2[keyword][-4] + df2[keyword][-3] + df2[keyword][-2]) / 3)
            trend3 = int((df2[keyword][-3] + df2[keyword][-2] + df2[keyword][-1]) / 3)

            if trend3 > trend2 and trend2 > trend1:
                trend = 'UP'
            elif trend3 < trend2 and trend2 < trend1:
                trend = 'DOWN'
            else:
                trend = 'FLAT'
            frequency = df2[keyword].sum()
        except:
            trend = 'N/A'
            frequency = 0

    # Append the result to the result_df using pd.concat
    new_row = pd.DataFrame({'Keyword': [keyword], 'Trend': [trend], 'Frequency': [frequency], 'Image': [image_url]})
    result_df = pd.concat([result_df, new_row], ignore_index=True)

    time.sleep(pause)

# Ensure 'Frequency' is numeric
result_df['Frequency'] = pd.to_numeric(result_df['Frequency'])

# Display the result dataframe
st.subheader("Keyword Search Trends")
st.dataframe(result_df)

# Display global stats
up_count = len(result_df[result_df['Trend'] == 'UP'])
down_count = len(result_df[result_df['Trend'] == 'DOWN'])
flat_count = len(result_df[result_df['Trend'] == 'FLAT'])
na_count = len(result_df[result_df['Trend'] == 'N/A'])
total_count = len(result_df)

st.write("Up: " + str(up_count) + " | " + str(round((up_count / total_count) * 100, 0)) + "%")
st.write("Down: " + str(down_count) + " | " + str(round((down_count / total_count) * 100, 0)) + "%")
st.write("Flat: " + str(flat_count) + " | " + str(round((flat_count / total_count) * 100, 0)) + "%")
st.write("N/A: " + str(na_count) + " | " + str(round((na_count / total_count) * 100, 0)) + "%")

# Plot the bar graph for top 10 maximum frequency keywords
top_10_keywords = result_df.nlargest(10, 'Frequency')
st.subheader("Top 10 Maximum Frequency Keywords")
st.bar_chart(top_10_keywords.set_index('Keyword')['Frequency'])

# Plot the line chart for time-wise interest of top three keywords
st.subheader("Time-wise Interest of Top Three Keywords")
top_3_keywords = result_df.nlargest(3, 'Frequency')['Keyword'].tolist()
line_chart_data = pd.DataFrame()

for keyword in top_3_keywords:
    retries = 3
    success = False
    while retries > 0 and not success:
        try:
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo, gprop='')
            df2 = pytrends.interest_over_time()
            if not df2.empty:
                df2 = df2.rename(columns={keyword: f"{keyword} Interest"})
                line_chart_data = pd.concat([line_chart_data, df2[f"{keyword} Interest"]], axis=1)
            success = True
        except TooManyRequestsError:
            retries -= 1
            time.sleep(pause * 5) 

if not line_chart_data.empty:
    st.line_chart(line_chart_data)

# Display trending fashion items
st.subheader("Trending Fashion")

top_10_images = top_10_keywords[['Keyword', 'Image']].reset_index(drop=True)

for i in range(0, len(top_10_images), 5):
    cols = st.columns(5, gap="large")
    for j in range(5):
        if i + j < len(top_10_images):
            try:
                cols[j].image(top_10_images.iloc[i + j]['Image'], caption=f"{top_10_images.iloc[i + j]['Keyword']}", use_column_width=True)
            except Exception as e:
                cols[j].write(f"Image not found for {top_10_images.iloc[i + j]['Keyword']}")
    st.markdown("<br>", unsafe_allow_html=True) 








