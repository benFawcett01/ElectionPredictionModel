import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import stats
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

                date = str(row['Year']) + month + day
            elif '/' in row['Published']:
                day = row['Published'][0:1]
                month = "0" + row['Published'][2:3]
                date = str(row['Year']) + month + day
            else:
                date = str(row['Year']) + str(row['Month']) + str(row['Published'])

            if len(date) < 8:
                date = date[0:6] + "0" + date[6]

            dates.append(date)

        date_df['date'] = dates

        new_df.drop(['Year', 'Month', 'Fieldwork', 'Published', 'Ukip'], axis=1, inplace=True)
        new_df['date_published'] = dates

        new_df['date_published'] = pd.to_datetime(new_df['date_published'], format='%Y%m%d')

        ordinals = []
        for i in new_df['date_published']:
            ordinals.append(i.toordinal())

        new_df['date_ordinals'] = ordinals

        new_df.to_csv("datasets/CleanOpinionPolls.csv", index=False)


    def pre_process_pollbase(self):
        clean_df = pd.read_csv('datasets/CleanOpinionPolls.csv', index_col=0)
        clean_df['date_published'] = pd.to_datetime(clean_df['date_published'], errors='coerce')
        clean_df.sort_values(by="date_ordinals", inplace=True)


        clean_df.dropna(inplace=True)
        # Deleting outliers

        for i in clean_df.columns:
            if not "date" in i:
                upper = clean_df[i].mean() + clean_df[i].std()
                lower = clean_df[i].mean() - clean_df[i].std()
                clean_df = clean_df.loc[(clean_df[i] < upper) & (clean_df[i] > lower)]

        clean_df.to_csv('datasets/PreprocessedOpinionPolls.csv', index=False)

    def forecast_popular_vote(self):
        model = Ridge()
        pp_df = pd.read_csv('datasets/PreprocessedOpinionPolls.csv')

        model_results ={
            "Polling": ["model"],
            "Con":[],
            "Lab":[],
            "LD":[],
            "Green":[],
            "BXP/Reform":[],
            "date_published":["2024-12-08"],
            "date_ordinals":[pd.to_datetime("2024-12-08").toordinal()]
        }

        print(pp_df.columns)

        for i in pp_df.columns:
            if not "date" in i:
                x = pp_df['date_ordinals'].tail(100).values.reshape(-1, 1)
                y = pp_df[i].tail(100).values.reshape(-1, 1)
                pred_date = "2024-12-08"
                pred_date = pd.to_datetime(pred_date)
                pred_date_ords = pred_date.toordinal()

                model.fit(x, y)
                pred = model.predict([[pred_date_ords]])
                model_results[i].append(round(pred[0][0], 1))


        print(model_results)

        model_results = pd.DataFrame(model_results)
        clean_df = pd.read_csv("datasets/CleanOpinionPolls.csv")
        pp_df = pd.concat([pp_df, model_results.iloc[:, 1:]], ignore_index=True)
        clean_df = pd.concat([clean_df, model_results], ignore_index=True)
        pp_df.to_csv("datasets/PreprocessedOpinionPolls.csv")
        clean_df.to_csv("datasets/CleanOpinionPolls.csv")

    def gen_popvote_graph(self):
        clean_df = pd.read_csv("datasets/CleanOpinionPolls.csv")
        pp_df = pd.read_csv("datasets/PreprocessedOpinionPolls.csv")

        clean_df = clean_df.iloc[:, 2:]
        colours = ["blue", "red", "orange", "green", "cyan"]
        for i, c in zip(clean_df.columns, colours):
            if not "date" in i:
                x = pp_df["date_ordinals"]
                y = pp_df[i]
                ax = sns.scatterplot(data=clean_df, y=i, s=5, x="date_ordinals", color=c)
                m, b = np.polyfit(x, y, 1)
                ax.plot(x, m*x+b, color = c, label=i)
                plt.annotate(str(pp_df[i][len(pp_df)-1]), xy=(pp_df["date_ordinals"][len(pp_df)-1]+90, pp_df[i][len(pp_df)-1]))

        plt.legend(fontsize="7")
        ax.set_xticklabels(pp_df["date_published"])
        plt.xticks(rotation=30, fontsize=7)
        ax.set(xlabel='Date', ylabel='% Vote')
        plt.savefig("static/opinionpollgraph.png", dpi=250)
        print("done")
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
                rankAv = rankAv + sec_df['ranking_total'][i]
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

        new_df.sort_values(by='Rank', inplace=True)

        #save clean sec dataset to dataset directory
        new_df.to_csv("datasets/clean_sec.csv")


    def clean_historical(self):

        # Get excel file instance of the historical dataset
        file = pd.ExcelFile('datasets/election_history_cut_nospaces.xlsx')
        df_list = []

        #Iterate through all the sheets in the historical dataset and clean them using clean_history_sheet()
        for i in file.sheet_names:
            x = self.clean_historical_sheet(pd.read_excel('datasets/election_history_cut_nospaces.xlsx', i))
            df_list.append(x)
            string = "datasets/historical_csv/" + i + ".csv"
            x.to_csv(string)




        # Create a new CSV for each party, showing their electoral history in each constituency
        csv_names = [col for col in df_list[len(df_list)-1] if 'Vote' in col]

        party_dfs = []

        print(csv_names)

        df = pd.DataFrame({
            'id':[], 'Constituency':[], '1964 Vote Share':[], '1966 Vote Share':[], '1970 Vote Share': [],
            '1974 (F) Vote Share': [], '1974 (O) Vote Share': [], '1979 Vote Share': [], '1983 Vote Share': [],
            '1987 Vote Share': [], '1992 Vote Share': [], '1997 Vote Share': [], '2001 Vote Share': [],
            '2005 Vote Share': [], '2010 Vote Share': [], '2015 Vote Share': [], '2017 Vote Share': [],
            '2019 Vote Share': [],
        })

        df['id'] = df_list[len(df_list)-1]['id']
        df['Constituency'] = df_list[len(df_list)-1]['Constituency']




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

        return sheet #return the cleaned sheet




if __name__ == "__main__":
    m = model()
    m.clean_pollbase()
    m.pre_process_pollbase()
    m.forecast_popular_vote()
    m.gen_popvote_graph()


