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
        #for example GOV-1001, WES-1001
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
        
        #generate financial data
        income = int(np.random.gamma(shape=2.0, scale=income_base/2.0)) #gamma for realistic wealth skew
        #cap to avoid unrealistic outliers
        income = max(8000, min(income, 250000))
        
        #arrears - using income and area to generate risk
        arrears = 0
        # If random roll is less than the area's risk probability...
        if np.random.random() < arrears_prob: 
            #lower income households tend to have higher/harder debt
            if income < (income_base * 0.8):
                arrears = np.random.randint(500, 3500)
            else:
                arrears = np.random.randint(100, 1000)
                
        #setting target to non-claimants of benefits
        claiming = 'No'
        #logic for benefit claims - income under 19000 eligible for housing assistance - will go more in-depth later
        #randomness: setting 40% of them haven't claimed yet (the "vulnerable" group i'm identifying)
        if income < 19000:
            if np.random.random() > 0.4: 
                claiming = 'Yes'

        #append the data
        data.append([hh_id, sector, income, arrears, claiming, lat, lon])

    #save to csv file
    df = pd.DataFrame(data, columns=['Household_ID', 'Postcode', 'Income', 'Arrears', 'Claiming_Benefits', 'lat', 'lon'])
    df.to_csv(filename, index=False)
    print(f"✅ Saved {filename} with {len(df)} records.")

#setting council zones - wanted to create 3 initial csv files that can be used to examine map data and make sure the map works
#three different council zones with large wage disparity 

# 1. GOVAN (Glasgow) - high deprivation, smaller dataset
# postcodes: G51 1 (Ibrox/Govan), G51 2 (Pacific Quay), G51 3 (Drumoyne)
generate_zone_data(
    filename="govan_data.csv",
    area_name="Govan",
    postcode_sectors=["G51 1", "G51 2", "G51 3"],
    n_households=500,
    income_base=21000, #lower average income
    arrears_prob=0.35  #higher risk of debt
)

# 2. LEITH (Edinburgh) - mixed demographics, medium dataset
# Postcodes: EH6 4 (Newhaven), EH6 6 (Leith Walk) (best place in edinburgh, my birthplace), EH6 7 (Restalrig)
generate_zone_data(
    filename="leith_data.csv",
    area_name="Leith",
    postcode_sectors=["EH6 4", "EH6 6", "EH6 7"],
    n_households=650,
    income_base=32000, #mid-range income
    arrears_prob=0.20  #moderate risk
)

# 3. WESTMINSTER (London) - high income, large dataset
# Postcodes: SW1V 3 (Pimlico), SW1P 4 (Westminster) (boo), SW1E 6 (Victoria)
generate_zone_data(
    filename="westminster_data.csv",
    area_name="Westminster",
    postcode_sectors=["SW1V 3", "SW1P 4", "SW1E 6"],
    n_households=750,
    income_base=55000, #high average income
    arrears_prob=0.10  #lower risk (but debts might be larger value due to adjusted avg income)
)