from ydata_profiling import ProfileReport
import pandas as pd

# Load the data
df = pd.read_csv('Data/onlinefoods.csv')
df = pd.DataFrame(df)

# Generate the report
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile.to_file("EDA_Report_foods.html")
# Open the report in the browser
#Just drag and drop the EDA_Report.html file to the browser


