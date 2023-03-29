import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

        monthDict = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Jul':'07', 'Aug':'08',
                     'Sep':'09', 'Oct':'10','Nov':'11', 'Dec':'12'}

        date_df['Month'] = date_df['Month'].astype("string")
        date_df['Month'].str.strip()
        date_df['Month'] = date_df['Month'].map(monthDict)

        date_df['Month'] = date_df['Month'].fillna(0)

        for index, row in date_df.iterrows():
            if '-' in row['Published']:
                day = row['Published'][0:1]
                month = row['Published'][3:6]

                date = day + month + str(row['Year'])
            elif '/' in row['Published']:
                day = row['Published'][0:1]
                month = "0" + row['Published'][2:3]
                date = day + month + str(row['Year'])
            else:
                date = str(row['Published']) + str(row['Month']) + str(row['Year'])

            if len(date) < 8:
                date = "0"+date

            dates.append(date)

        date_df['date'] = dates

        for i in date_df['date']:
            if len(i) < 8:
                print(i)

        new_df.drop(['Year', 'Month', 'Fieldwork', 'Published', 'Ukip'], axis=1, inplace=True)
        new_df['date_published'] = dates

        new_df['date_published'] = pd.to_datetime(new_df['date_published'], format='%d%m%Y')

        count = 8


        new_df.to_csv("datasets/CleanOpinionPolls.csv", index=False)


    def pre_process_pollbase(self):
        clean_df = pd.read_csv('datasets/CleanOpinionPolls.csv')
        clean_df = clean_df.iloc[:, 1:]
        clean_df['date_published'] = pd.to_datetime(clean_df['date_published'], errors='coerce')
        clean_df.sort_values(by='date_published', inplace=True)

        for i in clean_df['date_published']:
            print(i)

        ax = sns.scatterplot(data=clean_df, palette=['blue', 'red', 'orange', 'green', 'cyan'])

        plt.savefig("static/opinionpollgraph.png")



        print("figure saved")



    def clean_socioeconomic_data(self):
        sec_df = pd.read_csv("datasets/sec_csv.csv") # Turn sec.csv into a dataset
        new_df = pd.DataFrame()

        rankAv = 0
        count = 0
        totalrank = {'ConstituencyName': [], 'Rank': []}

        '''Iterate through all neighbourhoods in sec dataset, calculate average SEC rank for the constituency, add 
        totalrank dictionary'''
        for i in range(1, len(sec_df)):
            if sec_df['ConstituencyName'][i] == sec_df['ConstituencyName'][i-1]:
                rankAv = rankAv + sec_df['rank'][i]
                count = count + 1
            elif (sec_df['ConstituencyName'][i] != sec_df['ConstituencyName'][i-1]) or sec_df['ConstituencyName'][i] == "End":
                string = str(sec_df['ConstituencyName'][i-1]) + ": " + str((rankAv / count))
                totalrank['ConstituencyName'].append(sec_df['ConstituencyName'][i-1])
                totalrank['Rank'].append(rankAv/count)
                rankAv = sec_df['rank'][i]
                count = 1

        #new_df holds data from totalrank
        new_df['ConstituencyName'] = totalrank['ConstituencyName']
        new_df['Rank'] = totalrank['Rank']

        #save clean sec dataset to dataset directory
        new_df.to_csv("datasets/clean_sec.csv")


    def clean_historical(self):

        # Get excel file instance of the historical dataset
        file = pd.ExcelFile('datasets/election_history_cut_nospaces.xlsx')
        df_list = []

        #Iterate through all the sheets in the historical dataset and clean them using clean_history_sheet()
        for i in file.sheet_names:
            df_list.append(self.clean_historical_sheet(pd.read_excel('datasets/election_history_cut_nospaces.xlsx', i)))

        # Create a new CSV for each party, showing their electoral history in each constituency
        csv_names = [col for col in df_list[len(df_list)-1] if 'Vote' in col]
        df = pd.DataFrame({
            'id':[], 'Constituency':[], '1964 Vote Share':[], '1966 Vote Share':[], '1970 Vote Share': [],
            '1974 (F) Vote Share': [], '1974 (O) Vote Share': [], '1979 Vote Share': [], '1983 Vote Share': [],
            '1987 Vote Share': [], '1992 Vote Share': [], '1997 Vote Share': [], '2001 Vote Share': [],
            '2005 Vote Share': [], '2010 Vote Share': [], '2015 Vote Share': [], '2017 Vote Share': [],
            '2019 Vote Share': [],
        })

        ''' for i in csv_names:
            for j in df_list:
                j[]'''



    def clean_historical_sheet(self, sheet):

        headers = sheet.columns
        newheaders = []

        for i in range(0, len(headers)): #remove whitespace from column headers
            newheaders.append(headers[i].strip())

        sheet.set_axis(newheaders, axis=1, inplace=True) #replace column names with non-whitespace newheaders list

        sheet.drop(['County', 'Electorate', 'Turnout'], axis=1, inplace=True) #Drop County column

        # drop columns with names that contain 'Votes
        votes = [col for col in sheet.columns if 'Votes' in col]
        for i in votes:
            try:
                sheet.drop(i, axis=1, inplace=True)
            except:
                sheet.drop('Votes.1', axis=1, inplace=True)
                votes.pop(0)

        print(sheet.columns)
        return sheet #return the cleaned sheet




if __name__ == "__main__":
    m = model()
    m.pre_process_pollbase()