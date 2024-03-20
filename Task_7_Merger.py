import csv
import pandas as pd

def converter(FileName):
    FileName = FileName.replace("B", "Б-")
    g = FileName.find("a")
    if g != -1:
        return FileName[0:g]
    g = FileName.find("b")
    if g != -1:
        return FileName[0:g]
    return FileName

with (open('temp/StatisticResult.csv', newline='') as csvfile1,
      open('temp/Petrograph.csv', newline='') as csvfile2,
      open('temp/ResultTable.csv', 'w', encoding='UTF8', newline='') as f):
    writer = csv.writer(f)
    rows_stat = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    rows_petro = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))

    ColNames = rows_stat[0] + ["ПетрографТипы", "ТипыТектонитов", "%матрикса"]
    writer.writerow(ColNames)

    tb_petro = pd.DataFrame(rows_petro[1:], columns=rows_petro[0])
    tb_stat = pd.DataFrame(rows_stat[1:], columns=rows_stat[0])

    for row in tb_stat.values:
        FileName = converter(row[0])
        row2 = tb_petro[tb_petro["НомерОбразца"].isin([FileName])]
        C1 = (row2["ПетрографТипы"].values.tolist())[0]
        C2 = (row2["ТипыТектонитов"].values.tolist())[0]
        C3 = (row2["%матрикса"].values.tolist())[0]
        row2 = row.tolist() + [C1, C2, C3]
        writer.writerow(row2)

    """for FileName in tb_stat["Номер Образца"].values:
        FileName = converter(FileName)
        row = tb_petro[tb_petro["НомерОбразца"].isin([FileName])]
        C1 = (row["ПетрографТипы"].values.tolist())[0]
        C2 = (row["ТипыТектонитов"].values.tolist())[0]
        C3 = (row["%матрикса"].values.tolist())[0]
        G = [C1, C2, C3]
        print(G)"""