#IMPORTANT - initial data generation script completed by me, updated using assistance from copilot 
#columns made with copilot marked as such

import pandas as pd
import numpy as np
import pgeocode

#initialising geocoder for the UK
nomi = pgeocode.Nominatim('gb')

#code to generate zone data for council areas
def generate_zone_data(filename, area_name, postcode_sectors, n_households, income_base, arrears_prob):
    print(f"🏗️  Generating {area_name} ({n_households} households)...")
    np.random.seed(42) #keeps consistent data
    
    data = []
    
    for i in range(n_households):
        #houshold ID
        prefix = area_name[:3].upper()
        hh_id = f"{prefix}-{1000+i}"
        
        #assign real postcode location
        sector = np.random.choice(postcode_sectors)
        location = nomi.query_postal_code(sector)
        
        #skip postcode if pgeocode can't locate
        if pd.isna(location.latitude):
            continue
            
        #adding jitter to ensure dots aren't stacked
        lat = location.latitude + np.random.normal(0, 0.002)
        lon = location.longitude + np.random.normal(0, 0.003)
        
        # --- NEW COLUMNS GENERATION --- COPILOT
        
        # 1. Household Size (1 to 6 people)
        # Weighted to make 1-4 most common
        hh_size = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.25, 0.30, 0.20, 0.15, 0.08, 0.02])
        
        # 2. Tenure Length (Years)
        # Using exponential distribution because many people stay short term, fewer stay 20+ years
        tenure = np.random.exponential(scale=6.0)
        tenure = round(max(0.1, tenure), 1) # Ensure min is 0.1 years
        
        # 3. EPC Rating (Energy Efficiency)
        # Weighted distribution (C and D are most common in UK)
        epc_opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        epc_probs = [0.02, 0.08, 0.35, 0.35, 0.12, 0.06, 0.02]
        epc = np.random.choice(epc_opts, p=epc_probs)
        
        # --- END NEW COLUMNS ---

        #generate financial data
        income = int(np.random.gamma(shape=2.0, scale=income_base/2.0)) 
        income = max(8000, min(income, 250000))
        
        #arrears variable
        arrears = 0
        
        #adding risk factors to adjust debt levels
        #further correlation with debt in visualisations with the data
        current_risk_prob = arrears_prob
        
        #if EPC is bad, greater risk of debt due to higher bills
        if epc in ['F', 'G']:
            current_risk_prob += 0.15
            
        #if statement for if tenure is less than 1 year (accounting for moving & furniture costs, initial deposit etc)
        if tenure < 1.0:
            current_risk_prob += 0.10

        #if statement to see if random roll is lower than calculated risk probability, appends base_debt
        if np.random.random() < current_risk_prob: 
            #lower income households tend to have higher/harder debt
            base_debt = np.random.randint(100, 1000)
            
            if income < (income_base * 0.8):
                base_debt = np.random.randint(500, 3500)
            
            #add extra debt factors for further negatives i.e. high fuel costs, large household with a low income 
            if epc in ['F', 'G']:
                base_debt += np.random.randint(200, 600) #fuel debt
            
            if hh_size > 4 and income < 25000:
                base_debt += np.random.randint(300, 800) #overcrowding
                
            arrears = base_debt
                
        #setting target to non-claimants of benefits
        claiming = 'No'
        if income < 19000:
            if np.random.random() > 0.4: 
                claiming = 'Yes'

        #append the data (Added the 3 new variables to the end)
        data.append([hh_id, sector, income, arrears, claiming, lat, lon, hh_size, tenure, epc])

    #save to csv file (Added columns to header)
    df = pd.DataFrame(data, columns=['Household_ID', 'Postcode', 'Income', 'Arrears', 'Claiming_Benefits', 'lat', 'lon', 'Household_Size', 'Tenure_Length_Years', 'EPC_Rating'])
    df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(df)} records.")

#setting council zones
# 1. GOVAN (Glasgow)
generate_zone_data(
    filename="govan_data.csv",
    area_name="Govan",
    postcode_sectors=["G51 1", "G51 2", "G51 3"],
    n_households=500,
    income_base=21000, 
    arrears_prob=0.35 
)

# 2. LEITH (Edinburgh) 
generate_zone_data(
    filename="leith_data.csv",
    area_name="Leith",
    postcode_sectors=["EH6 4", "EH6 6", "EH6 7"],
    n_households=650,
    income_base=32000, 
    arrears_prob=0.20 
)

# 3. WESTMINSTER (London)
generate_zone_data(
    filename="westminster_data.csv",
    area_name="Westminster",
    postcode_sectors=["SW1V 3", "SW1P 4", "SW1E 6"],
    n_households=750,
    income_base=55000, 
    arrears_prob=0.10 
)