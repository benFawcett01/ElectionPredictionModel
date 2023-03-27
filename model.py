import pandas as pd
import numpy as np
class model:

    running = False

    def clean_pollbase(self):
        new_df = pd.read_csv("datasets/opinionpolls.csv") # Read excel file

        # Noticed that all unnamed columns did not contain necessary data, t.f. drop
        new_df = new_df.loc[:, ~new_df.columns.str.contains('^Unnamed')] #Delete unnamed columns

        new_df = new_df[new_df['Con'].notna()] #drop polls that do not hold voting intention for conservatives
        new_df = new_df[new_df['Published'].notna()] #drop polls that do not have a published date
        new_df = new_df[new_df['Month'].notna()] #drop polls that do not have the month published

        new_df.Month = new_df['Month'].str.strip() #delete whitespace in strings in month column

        new_df.iloc[0, 0] = "2020"

        date_df = new_df.iloc[:, 0:4] #Used for formatting published date
        date_df['Year'] = pd.to_numeric(date_df['Year'], errors='coerce')
        date_df['Year'] = date_df['Year'].fillna(0)
        date_df['Year'] = date_df['Year'].astype(int)

        days = []
        month = 0
        months = []
        date = ''
        dates = []

        for index, row in date_df.iterrows():
            if '-' in row['Published']:
                days.append(row['Published'][0:1])
                months.append(row['Published'][3:6])
            else:
                days.append(row['Published'])
                months.append(row['Month'])

        date_df.drop(['Month'], axis=1)
        date_df.drop(['Published'], axis=1)

        date_df['Month'] = months
        date_df['Published'] = days

        monthDict = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10,
                     'Nov':11, 'Dec':12}

        date_df['Month'].str.strip()
        date_df['Month'] = date_df['Month'].map(monthDict)
        date_df['Month'] = date_df['Month'].fillna(0)
        date_df['Month'] = date_df['Month'].astype(int)
        for index, row in date_df.iterrows():
            if '-' in row['Published']:
                day = row['Published'][0:1]
                month = row['Published'][3:6]
                date = day + "." + month + "." + str(row['Year'])
            elif '/' in row['Published']:
                day = row['Published'][0:1]
                month = row['Published'][2:3]
                date = day + "." + month + "." + str(row['Year'])
            else:
                date = str(row['Published']) + "." + str(row['Month']) + "." + str(row['Year'])
            dates.append(date)

        date_df['date'] = dates

        #print(date_df)

        new_df.drop(['Year', 'Month', 'Fieldwork', 'Published', 'Ukip'], axis=1, inplace=True)
        new_df['date_published'] = dates

        new_df.to_csv("datasets/CleanOpinionPolls.csv", index=False)
        print(new_df)



if __name__ == "__main__":
    hello = model()
    hello.clean_pollbase()