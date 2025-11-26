s import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt #hate altair but necessary for colour schemes on Streamlit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    required_columns = ['Income', 'Arrears', 'Claiming_Benefits', 'lat', 'lon']
    missing = [col for col in required_columns if col not in df.columns]
    return missing

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def run_kmeans_clustering(df, n_clusters=3):
    #creating a debt-burden feature
    #using arrears and income to establish an arrears ratio, more accurately for k-means clustering 
    #high-income households might have high debt (but very low arrears ratio) so using this new value instead 
    df['Arrears_Ratio'] = df['Arrears'] / (df['Income'] + 1) #add 1 to income to avoid division by zero errors
    
    #selecting features for the k-means clustering
    #previously used income and arrears, changed to arrears_ratio to make more accurate clustering
    features = df[['Income', 'Arrears_Ratio']]
    
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
        df = pd.read_csv("govan_data.csv")
    except:
        #if my data generation script hasn't been run yet
        #will update this to have basic data on home screen instead of blank screen/errors
        st.sidebar.warning("⚠️ 'govan_data.csv' not found. Please run the generator script or upload a CSV file.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.header("2. Policy Parameters")
st.sidebar.caption("Adjust thresholds to filter the target households.")
income_threshold = st.sidebar.slider("Max Income Threshold (£)", 0, 80000, 18000)
arrears_threshold = st.sidebar.slider("Min Arrears Threshold (£)", 0, 2500, 500)

#filtering the data set by parameters
vulnerable_cohort = df[
    (df['Income'] < income_threshold) & 
    (df['Arrears'] > arrears_threshold) & 
    (df['Claiming_Benefits'] == 'No')
]

#DASHBOAR
#header section
header_col, metric_col1, metric_col2, metric_col3 = st.columns([2, 1, 1, 1])

with header_col:
    st.title("Detecting Struggling Households")
    st.caption(f"Analysing household data for council debt & unclaimed support eligibility.")

#spacer because the whitespace and placement was ridiculous
spacer_html = "<div style='height: 1.5rem;'></div>"

with metric_col1:
    st.markdown(spacer_html, unsafe_allow_html=True) 
    st.metric("Risk Group Size:", len(vulnerable_cohort))
    
with metric_col2:
    st.markdown(spacer_html, unsafe_allow_html=True)
    #rough estimation for potential savings i.e. child tax credits
    potential_saving = len(vulnerable_cohort) * 1500 
    st.metric("Est. Unclaimed Support", f"£{potential_saving:,}")
    
with metric_col3:
    st.markdown(spacer_html, unsafe_allow_html=True)
    avg_arrears = vulnerable_cohort['Arrears'].mean() if not vulnerable_cohort.empty else 0
    st.metric("Avg Arrears (Group)", f"£{avg_arrears:.0f}")

st.write("") 

#splitting it up into tabs
#reduced unnecessary scrolling, neater design
tab1, tab2 = st.tabs(["Map View", "K-Means Analysis"])

# MAP SECTION!!!
with tab1:
    if not vulnerable_cohort.empty:
        col_map, col_list = st.columns([3, 1])
        
        with col_map:
            st.subheader("Housing Data Distribution")
            
            #dynamic tooltip, checks for postcode then displays if it exists
            tooltip_html = "<b>ID:</b> {Household_ID}<br/><b>Arrears:</b> £{Arrears}"
            if 'Postcode' in vulnerable_cohort.columns:
                tooltip_html += "<br/><b>Postcode:</b> {Postcode}"

            layer = pdk.Layer(
                "ScatterplotLayer",
                vulnerable_cohort,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=25, #smaller dot size for higher accuracy on the map
                pickable=True
            )
            
            #auto-zoom function to match csv data
            mid_lat = vulnerable_cohort['lat'].mean()
            mid_lon = vulnerable_cohort['lon'].mean()
            view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=11.5)
            
            st.pydeck_chart(pdk.Deck(
                layers=[layer], 
                initial_view_state=view_state,
                tooltip={"html": tooltip_html}
            ), height=400, use_container_width=True)
            
        #list for priority contact list
        with col_list:
            st.subheader("Priority List")
            export_df = vulnerable_cohort.sort_values(by='Arrears', ascending=False)
            
            #clean up columns for display
            display_cols = ['Household_ID', 'Arrears']
            if 'Postcode' in export_df.columns:
                display_cols.append('Postcode')
                
            st.dataframe(export_df[display_cols], height=350, use_container_width=True)
            
            #download button for list of priority households to contact
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
    cluster_summary = df_clustered.groupby('Cluster')['Arrears_Ratio'].mean().sort_values()
    labels = ["Low Risk", "Moderate Risk", "High Risk"]
    risk_mapping = {id: label for id, label in zip(cluster_summary.index, labels)}
    df_clustered['Risk_Profile'] = df_clustered['Cluster'].map(risk_mapping)

    #column layout
    col_chart, col_stats = st.columns([2, 1]) 
    
    with col_chart:
        st.subheader("K-Means Clustering")
        st.caption("Households grouped by Debt-to-Income Ratio to examine their financial burden.")
        
        #custom colours for chart
        #defined colour mapping: high risk = red, moderate = orange, low = green
        domain = ["High Risk", "Moderate Risk", "Low Risk"]
        range_ = ["#ff0000", "#ffa500", "#008000"]
        
        #reset_index to ensure the chart draws correctly
        chart_data = df_clustered.reset_index(drop=True)

        chart = alt.Chart(chart_data).mark_circle(size=60).encode(
            x='Income',
            y='Arrears',
            color=alt.Color('Risk_Profile', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(title="Risk Level")),
            tooltip=['Household_ID', 'Income', 'Arrears', 'Risk_Profile', alt.Tooltip('Arrears_Ratio', format='.1%', title='Debt Burden')]
        ).properties(
            height=380
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        
    with col_stats:
        st.subheader("Cluster Statistics")
        
        summary_df = df_clustered.groupby('Risk_Profile')[['Income', 'Arrears', 'Arrears_Ratio']].mean()
        summary_df['Count'] = df_clustered['Risk_Profile'].value_counts()
        summary_df = summary_df.reset_index()

        #logic for sorting into risk categories for mapping
        custom_order = {"High Risk": 0, "Moderate Risk": 1, "Low Risk": 2}
        summary_df['sort_key'] = summary_df['Risk_Profile'].map(custom_order)
        summary_df = summary_df.sort_values('sort_key').drop('sort_key', axis=1)
        summary_df.columns = ['Risk Profile', 'Avg Income', 'Avg Debt', 'Avg Burden', 'Households']

        #multiplying the avg burden by 100 to display an easier-to-read percentage
        summary_df['Avg Burden'] = summary_df['Avg Burden'] * 100

        #colour styling for table 
        def highlight_risk(val):
            if val == 'High Risk':
                return 'background-color: #ffcccc; color: #8b0000; font-weight: bold' # Red
            elif val == 'Moderate Risk':
                return 'background-color: #fff4cc; color: #9c6500' # Orange
            elif val == 'Low Risk':
                return 'background-color: #d6f5d6; color: #006400' # Green
            return ''

        #apply the style to the 'Risk Profile' column
        styled_df = summary_df.style.map(highlight_risk, subset=['Risk Profile'])

        #smart table to display csv results
        st.dataframe(
            styled_df,
            height=150, 
            hide_index=True,
            use_container_width=True,
            column_config={
                "Avg Income": st.column_config.NumberColumn(format="£%d"),
                "Avg Debt": st.column_config.NumberColumn(format="£%d"),
                "Avg Burden": st.column_config.NumberColumn(format="%.1f%%", help="Arrears as % of Income"),
                "Households": st.column_config.ProgressColumn(
                    format="%d", 
                    min_value=0, 
                    max_value=int(summary_df['Households'].max()),
                ),
            }
        )
        
        #export the list of high risk households
        high_risk_df = df_clustered[df_clustered['Risk_Profile'] == 'High Risk']
        high_risk_csv = high_risk_df.to_csv(index=False).encode('utf-8')
        
        st.warning(f"⚠️ **{len(high_risk_df)} Households** identified as High Risk.")
        #download button
        st.download_button(
            label="📥 Export High Risk Households",
            data=high_risk_csv,
            file_name="high_risk_cluster.csv",
            mime="text/csv",
            use_container_width=True

        )

