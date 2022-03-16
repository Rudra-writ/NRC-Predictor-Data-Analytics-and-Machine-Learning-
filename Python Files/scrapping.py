import pandas as pd
list_head = ['JCAT', 'Satcat','Piece','Type','Name','PLName', 'LDate','Parent','SDate','Primary','DDate','Status Dest','Owner','State','Manufacturer','Bus','Motor','Mass','DryMass','TotMass','Length','Diamete','Span','Shape','ODate' ,'Perigee','Apogee','Inc','OpOrbitOQU', 'AltNames']
df = pd.read_csv('P:/SATLAUNCHLOG.txt' ,  names = list_head)
print(df.head(50))
df.to_excel('P:/scraped.xlsx')
