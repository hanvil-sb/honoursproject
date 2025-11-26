import pandas as pd
import numpy as np
import pgeocode
import random

#initialise geocode
nomi = pgeocode.Nominatim('gb')

def generate_advanced_council_data(filename, area_name, postcode_sectors, n_households, economic_profile):
    np.random.seed(42)
    random.seed(42)
    
    #initialising the dataset
    data = []
    
    for i in range(n_households):
        #identifiers and postcode locations
        hh_id = f"{area_name[:3].upper()}-{1000+i}"
        sector = np.random.choice(postcode_sectors)
        loc = nomi.query_postal_code(sector)
        if pd.isna(loc.latitude): continue
        lat = loc.latitude + np.random.normal(0, 0.002)
        lon = loc.longitude + np.random.normal(0, 0.003)
        
        #demographics (age, amount of children etc)
        age = int(np.random.normal(45, 15))
        age = max(18, min(age, 90))
        
        hh_size = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.28, 0.30, 0.18, 0.14, 0.07, 0.03])
        if hh_size == 1:
            n_children = 0
        else:
            n_children = np.random.randint(0, hh_size)
            
        #employment columns (student, retired, part-time, full-time etc)
        if age >= 67:
            emp_status = 'Retired'
        elif age < 22 and np.random.random() > 0.7:
            emp_status = 'Student'
        else:
            if np.random.random() < economic_profile['unemployment_rate']:
                emp_status = 'Unemployed'
            elif np.random.random() < 0.2: 
                emp_status = 'Part-time'
            else:
                emp_status = 'Full-time'
                
        #low chance of disability label        
        disability_chance = 0.10 + (0.005 * (age - 18))
        disability = 'Yes' if np.random.random() < disability_chance else 'No'

        #financial data
        base = economic_profile['base_income']
        
        #setting minimum requirements for income statuses
        #prevents a weird data skew at 0 income
        if emp_status == 'Unemployed':
            income = np.random.randint(3800, 8500) 
        elif emp_status == 'Student':
            income = np.random.randint(4000, 9000) # Loans/Grants
        elif emp_status == 'Retired':
            # State pension is approx 11k, plus private pension variation
            income = np.random.normal(13000, 5000) 
        elif emp_status == 'Part-time':
            income = np.random.normal(base * 0.6, 4000)
        else:
            #bell curve for incomes
            #ensures most cases meet the average UK wages
            income = np.random.normal(base, 9000)
            
        #hard boundaries to stop extreme outliers
        income = round(max(3800, min(income, 150000)))
        
        #savings column, important for understanding further debt
        savings = int(np.random.exponential(economic_profile['savings_skew']))
        
        #property columns (length of tenancy, beds, rent etc)
        tenure = round(np.random.exponential(6.0), 1)
        beds = max(1, min(5, hh_size + np.random.randint(-1, 2)))
        rent = 400 + (beds * 150) + np.random.randint(-50, 50)
        epc = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], p=[0.01, 0.05, 0.30, 0.40, 0.15, 0.07, 0.02])
        
        #benefits columns
        #uses attributes to append certain benefits to eligible column
        eligible_benefits = []
        if n_children > 0: eligible_benefits.append('Child Benefit')
        if hh_size >= 3 and n_children > 0 and income < 30000 and savings < 16000: eligible_benefits.append('Child Tax Credit')
        if hh_size == 1 and income < 18000 and savings < 16000: eligible_benefits.append('Universal Credit')
        elif income < 16000 and savings < 16000 and age < 67: eligible_benefits.append('Universal Credit')
        if disability == 'Yes': eligible_benefits.append('PIP')
        if age >= 67 and income < 12000: eligible_benefits.append('Pension Credit')

        actual_benefits = eligible_benefits.copy()
        if len(actual_benefits) > 0 and np.random.random() < 0.25:
            drop_idx = np.random.randint(0, len(actual_benefits))
            actual_benefits.pop(drop_idx)
            
        eligible_str = ", ".join(eligible_benefits) if eligible_benefits else "None"
        claimed_str = ", ".join(actual_benefits) if actual_benefits else "None"
        
        #generating a risk score for examples
        risk_score = 0
        if income < 18000: risk_score += 3
        if savings < 500: risk_score += 2
        if epc in ['F', 'G']: risk_score += 2 
        if beds > hh_size: risk_score += 2 
        if n_children >= 3: risk_score += 1
        if disability == 'Yes': risk_score += 1
        
        council_tax_arrears = 'No'
        if risk_score > 4 and np.random.random() < 0.6:
            council_tax_arrears = 'Yes'
            risk_score += 5 
            
        arrears = 0
        threshold = 5 
        volatility = np.random.normal(0, 2)
        
        if (risk_score + volatility) > threshold:
            base_debt = rent * np.random.uniform(0.5, 4.0)
            arrears = round(base_debt)
            
        data.append([
            hh_id, sector, lat, lon, 
            age, emp_status, hh_size, n_children, disability, 
            income, savings, rent, 
            tenure, beds, epc, 
            eligible_str, claimed_str, 
            council_tax_arrears, arrears 
        ])

    columns = [
        'Household_ID', 'Postcode', 'lat', 'lon',
        'Age', 'Employment_Status', 'Household_Size', 'Num_Children', 'Disability',
        'Annual_Income', 'Savings_Capital', 'Monthly_Rent',
        'Tenure_Years', 'Property_Beds', 'EPC_Rating',
        'Benefits_Eligible', 'Benefits_Claimed',
        'Council_Tax_Arrears', 'Rent_Arrears'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"✅ Created {filename} ({len(df)} records) with Normalized Income Distribution.")

#generating for different council zones
# 1. GOVAN 
govan_profile = {'base_income': 24000, 'unemployment_rate': 0.15, 'savings_skew': 2000}
generate_advanced_council_data("govan_data.csv", "Govan", ["G51 1", "G51 2"], 500, govan_profile)

# 2. LEITH 
leith_profile = {'base_income': 31000, 'unemployment_rate': 0.08, 'savings_skew': 6000}
generate_advanced_council_data("leith_data.csv", "Leith", ["EH6 4", "EH6 6"], 600, leith_profile)

# 3. WESTMINSTER 
west_profile = {'base_income': 48000, 'unemployment_rate': 0.04, 'savings_skew': 15000}
generate_advanced_council_data("westminster_data.csv", "Westminster", ["SW1V 3", "SW1P 4"], 700, west_profile)