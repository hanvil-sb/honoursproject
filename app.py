import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt #hate altair but necessary for colour schemes on Streamlit
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

#page config and layout 
st.set_page_config(
    page_title="Council Housing Insights", 
    layout="wide",
    initial_sidebar_state="expanded"
)

#basic css to help it fit the page
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.6rem;
        }
    </style>
""", unsafe_allow_html=True)

#data processing
def validate_columns(df):
    #setting standard schema for this application
    #stating columns required in csv file, will update as the project expands
    required_columns = [
        'Annual_Income', 'Rent_Arrears', 'Benefits_Claimed', 'Benefits_Eligible',
        'Savings_Capital', 'EPC_Rating', 'Household_Size', 'lat', 'lon'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    return missing

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    
    #cleaning up numeric columns to prevent errors
    cols_to_numeric = ['Annual_Income', 'Rent_Arrears', 'Savings_Capital', 'Household_Size']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    #parsing the benefit lists (converting string "['A', 'B']" to actual counts)
    if 'Benefits_Eligible' in df.columns and 'Benefits_Claimed' in df.columns:
        df['Eligible_Count'] = df['Benefits_Eligible'].apply(lambda x: len(str(x).split(',')) if str(x).lower() != 'none' else 0)
        df['Claimed_Count'] = df['Benefits_Claimed'].apply(lambda x: len(str(x).split(',')) if str(x).lower() != 'none' else 0)
        df['Unclaimed_Gap'] = df['Eligible_Count'] - df['Claimed_Count']
        #ensure no negative gaps (data errors)
        df['Unclaimed_Gap'] = df['Unclaimed_Gap'].clip(lower=0)
        
    return df

def run_kmeans_clustering(df, n_clusters=3):
    #creating a debt-burden feature
    #using arrears and income to establish an arrears ratio, more accurately for k-means clustering 
    #high-income households might have high debt (but very low arrears ratio) so using this new value instead 
    df['Debt_Burden'] = df['Rent_Arrears'] / (df['Annual_Income'] + 1) #add 1 to income to avoid division by zero errors
    
    #selecting features for the k-means clustering
    features = df[['Annual_Income', 'Debt_Burden']]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    return df

#sidebar for data uploads and setting parameters
st.sidebar.header("1. Data Integration")
uploaded_file = st.sidebar.file_uploader("Upload Council Data (CSV)", type=["csv"])

if uploaded_file is not None:
    #if statement catches if csv file is present
    try:
        df = load_data(uploaded_file)
        missing_cols = validate_columns(df)
        if missing_cols:
            st.error(f"Schema Error: Missing columns {', '.join(missing_cols)}")
            st.stop()
        st.sidebar.success("✅ Dataset Successfully Ingested")
    except Exception as e:
        st.error(f"File Error: {e}")
        st.stop()
else:
    #default council data set to Govan
    try:
        df = load_data("govan_data.csv")
    except:
        #if my data generation script hasn't been run yet
        #will update this to have basic data on home screen instead of blank screen/errors
        st.sidebar.warning("⚠️ 'govan_data.csv' not found. Please run the generator script or upload a CSV file.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.header("2. Policy Parameters")
st.sidebar.caption("Adjust thresholds to filter the target households.")
income_threshold = st.sidebar.slider("Max Income Threshold (£)", 0, 80000, 25000)
arrears_threshold = st.sidebar.slider("Min Arrears Threshold (£)", 0, 2500, 500)

#filtering the data set by parameters
vulnerable_cohort = df[
    (df['Annual_Income'] < income_threshold) & 
    (df['Rent_Arrears'] > arrears_threshold)
]

#DASHBOAR
#header section
header_col, metric_col1, metric_col2, metric_col3 = st.columns([2, 1, 1, 1])

with header_col:
    st.title("Detecting Struggling Households")
    st.caption("Advanced Analytics for Council Housing Debt & Support Eligibility.")

#spacer because the whitespace and placement was ridiculous
spacer_html = "<div style='height: 1.5rem;'></div>"

with metric_col1:
    st.markdown(spacer_html, unsafe_allow_html=True) 
    st.metric("Risk Group Size:", len(vulnerable_cohort))
    
with metric_col2:
    st.markdown(spacer_html, unsafe_allow_html=True)
    avg_arrears = vulnerable_cohort['Rent_Arrears'].mean() if not vulnerable_cohort.empty else 0
    st.metric("Avg Arrears (Group)", f"£{avg_arrears:.0f}")

with metric_col3:
    st.markdown(spacer_html, unsafe_allow_html=True)
    #calculating total missing claims in this group
    unclaimed_total = vulnerable_cohort['Unclaimed_Gap'].sum() if 'Unclaimed_Gap' in vulnerable_cohort.columns else 0
    st.metric("Unclaimed Support Cases", int(unclaimed_total), help="Total number of missing benefit claims in this group")

st.write("") 

#splitting it up into tabs
#reduced unnecessary scrolling, neater design
tab1, tab2, tab3 = st.tabs(["🌍 Map View", "K-Means Clusters", "Predictive Modelling"])

# MAP SECTION!!!
with tab1:
    if not vulnerable_cohort.empty:
        col_map, col_list = st.columns([3, 1])
        
        with col_map:
            st.subheader("Housing Data Distribution")
            
            #dynamic tooltip, checks for postcode then displays if it exists
            tooltip_html = "<b>ID:</b> {Household_ID}<br/><b>Arrears:</b> £{Rent_Arrears}<br/><b>Income:</b> £{Annual_Income}"
            
            layer = pdk.Layer(
                "ScatterplotLayer",
                vulnerable_cohort,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=30, #smaller dot size for higher accuracy on the map
                pickable=True
            )
            
            #auto-zoom function to match csv data
            mid_lat = vulnerable_cohort['lat'].mean()
            mid_lon = vulnerable_cohort['lon'].mean()
            view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=12)
            
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"html": tooltip_html}), height=400)
            
        #list for priority contact list
        with col_list:
            st.subheader("Priority List")
            
            export_df = vulnerable_cohort.sort_values(by='Rent_Arrears', ascending=False)
            #clean up columns for display
            display_cols = ['Household_ID', 'Rent_Arrears', 'Unclaimed_Gap']
            
            st.dataframe(export_df[display_cols], height=350, use_container_width=True)

            #download button to download csv of priority outreach list
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Priority Outreach List",
                data=csv,
                file_name="priority_outreach_list.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("No households match the current risk criteria.")

#K-means clustering
with tab2:
    #k-means clustering df
    df_clustered = run_kmeans_clustering(df.copy(), n_clusters=3)
    
    #automatically label clusters based on ratio (Burden), not just amount
    #high ratio = high risk
    cluster_summary = df_clustered.groupby('Cluster')['Debt_Burden'].mean().sort_values()
    labels = ["Low Risk", "Moderate Risk", "High Risk"]
    risk_mapping = {id: label for id, label in zip(cluster_summary.index, labels)}
    df_clustered['Risk_Profile'] = df_clustered['Cluster'].map(risk_mapping)
    
    #column layout
    col_k_chart, col_k_data = st.columns([2, 1])
    
    with col_k_chart:
        st.subheader("K-Means Clustering")
        st.caption("Households grouped by Debt-to-Income Ratio To Examine At-Risk Households.")
        
        #custom colours for chart
        #defined colour mapping: high risk = red, moderate = orange, low = green
        domain = ["High Risk", "Moderate Risk", "Low Risk"]
        range_ = ["#ff0000", "#ffa500", "#008000"]

        #reset_index to ensure the chart draws correctly
        chart_data = df_clustered.reset_index(drop=True)
        
        chart = alt.Chart(chart_data).mark_circle(size=60).encode(
            x=alt.X('Annual_Income', title='Annual Income (£)'),
            y=alt.Y('Rent_Arrears', title='Rent Arrears (£)'),
            color=alt.Color('Risk_Profile', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(title="Risk Level")),
            tooltip=['Household_ID', 'Annual_Income', 'Rent_Arrears', 'Risk_Profile']
        ).properties(height=380).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
    with col_k_data:
        st.subheader("Cluster Statistics")
        
        #getting the means
        summary = df_clustered.groupby('Risk_Profile')[['Annual_Income', 'Rent_Arrears', 'Debt_Burden']].mean()
        #adding household counts
        summary['Households'] = df_clustered['Risk_Profile'].value_counts()
        summary = summary.reset_index()

        # Sort by risk
        custom_order = {"High Risk": 0, "Moderate Risk": 1, "Low Risk": 2}
        summary['sort_key'] = summary['Risk_Profile'].map(custom_order)
        summary = summary.sort_values('sort_key').drop('sort_key', axis=1)
        
        #rename for display
        summary.columns = ['Risk Profile', 'Avg Income', 'Avg Debt', 'Avg Burden', 'Households']
        #multiplying the avg burden by 100 to display it as a percentage instead of a fraction
        summary['Avg Burden'] = summary['Avg Burden'] * 100

        #colour styling for table 
        def highlight_risk(val):
            if val == 'High Risk': return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
            elif val == 'Moderate Risk': return 'background-color: #fff4cc; color: #9c6500'
            elif val == 'Low Risk': return 'background-color: #d6f5d6; color: #006400'
            return ''

        styled_df = summary.style.map(highlight_risk, subset=['Risk Profile'])

        #smart table to display results
        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Avg Income": st.column_config.NumberColumn(format="£%.0f"),
                "Avg Debt": st.column_config.NumberColumn(format="£%.0f"),
                "Avg Burden": st.column_config.NumberColumn(format="%.0f%%", help="Debt as % of Income"),
                "Households": st.column_config.ProgressColumn(
                    format="%d", 
                    min_value=0, 
                    max_value=int(summary['Households'].max()),
                ),
            }
        )
        
        #download button for csv file of high risk households
        high_risk_df = df_clustered[df_clustered['Risk_Profile'] == 'High Risk']
        high_risk_csv = high_risk_df.to_csv(index=False).encode('utf-8')
        
        st.write("") 
        st.warning(f"⚠️ **{len(high_risk_df)} Households** identified as High Risk.")
        st.download_button(
            label="📥 Download High Risk Households Contact List",
            data=high_risk_csv,
            file_name="high_risk_cluster.csv",
            mime="text/csv",
            use_container_width=True
        )

#random forest model section
with tab3:
    st.subheader("Predictive Modelling: Why do tenants fall into debt?")
    st.caption("This model uses Random Forest Classifier on the provided dataset to identify the strongest predictors of financial issues.")

    #data prep for the model
    ml_df = df.copy()
    
    #encode categorical text data into numbers the model can understand
    le = LabelEncoder()
    #handle EPC
    epc_map = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
    ml_df['EPC_Score'] = ml_df['EPC_Rating'].map(epc_map).fillna(4)
    ml_df['Disability_Encoded'] = le.fit_transform(ml_df['Disability'].astype(str))
    
    #define Target: High Risk (> £500 debt)
    ml_df['Is_High_Risk'] = (ml_df['Rent_Arrears'] > 500).astype(int)
    
    #selecting features for the model
    #including 'Savings' and 'Unclaimed_Gap' from dataset
    features = ['Annual_Income', 'Savings_Capital', 'EPC_Score', 'Household_Size', 'Unclaimed_Gap', 'Disability_Encoded']
    
    X = ml_df[features].fillna(0)
    y = ml_df['Is_High_Risk']
    
    #training the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    #visualize importance
    importances = pd.DataFrame({
        'Factor': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    col_ai_chart, col_ai_insight = st.columns([2, 1])
    
    with col_ai_chart:
        #altair bar chart for feature importance
        bar_chart = alt.Chart(importances).mark_bar().encode(
            x=alt.X('Importance', axis=alt.Axis(format='%'), title='Predictive Power'),
            y=alt.Y('Factor', sort='-x', title='Risk Factor'),
            color=alt.Color('Importance', scale=alt.Scale(scheme='tealblues')),
            tooltip=['Factor', alt.Tooltip('Importance', format='.1%')]
        ).properties(height=350)
        st.altair_chart(bar_chart, use_container_width=True)

    with col_ai_insight:
        top_factor = importances.iloc[0]['Factor']
        top_score = importances.iloc[0]['Importance']
        
        st.info(f"💡 **Key Finding:** The model has identified **{top_factor}** as the single strongest predictor of debt (Influence: {top_score:.1%}).")
        
        #specific narrative based on what is likely to be top
        #will expand/change as dataset is developed
        if top_factor == 'Savings_Capital':
            st.markdown("⚠️ **The Savings Trap:** Tenants with low savings buffer are falling into debt immediately when shocks occur.")
        elif top_factor == 'Unclaimed_Gap':
            st.markdown("⚠️ **System Failure:** The strong link between 'Unclaimed Gap' and debt proves that **administrative barriers** to claiming benefits are directly causing rent arrears.")
        elif top_factor == 'Annual_Income':
            st.markdown("⚠️ **Poverty Driver:** Low income is the primary driver, suggesting a need for employment support rather than just debt management.")
        
        