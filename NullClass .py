#!/usr/bin/env python
# coding: utf-8

# # TASK 1

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


apps_df=pd.read_csv("C:\\Users\\Sananya Roy\\Downloads\\Play Store Data (1).csv")
reviews_df=pd.read_csv("C:\\Users\\Sananya Roy\\Downloads\\User Reviews.csv")


# In[4]:


merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')
merged_df


# In[5]:


filtered_df = merged_df[(merged_df['Category'] == 'HEALTH_AND_FITNESS')]
filtered_df


# In[6]:


filtered_df.dropna(subset=['Sentiment'],inplace=True)


# In[7]:


filtered1_df=filtered_df[(filtered_df['Sentiment']=='Positive')]
filtered1_df


# In[8]:


text = " ".join(review for review in filtered1_df['Translated_Review'].dropna())
text


# In[9]:


from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.graph_objects as go
import re
import nltk
from nltk.corpus import stopwords
import os
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
text_data = " ".join(filtered1_df['Translated_Review'].astype(str).tolist())
text_data = re.sub(r"[^a-zA-Z\s]", "", text_data)
words = text_data.lower().split()
app_names = set(filtered1_df['App'].str.lower())
words_tokens = [word for word in words if word not in stop_words and word not in app_names and len(word) > 2]
word_freq = Counter(words_tokens)
word_freq_dict = dict(word_freq.most_common(1000))
wordcloud = WordCloud(width=400, height=300, background_color='white',prefer_horizontal=1.0, colormap='inferno').generate_from_frequencies(word_freq_dict)
buf = BytesIO()
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig(buf, format='png')
plt.close()
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')
buf.close()
fig1 = go.Figure()
fig1.add_layout_image(dict(source=f"data:image/png;base64,{img_base64}",xref="paper", yref="paper",x=0, y=1, sizex=1, sizey=1,xanchor="left", yanchor="top",sizing="stretch"))
fig1.update_layout(title="Word Cloud from 5-Star Reviews (Health & Fitness)",plot_bgcolor='white',paper_bgcolor='white',font_color='white',title_font={'size':16},xaxis=dict(title_font={'size':12}),yaxis=dict(title_font={'size':12}),margin=dict(l=6,r=6,t=10,b=5))
fig1.write_html("dashboard.html")


# # TASK 2

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


df=pd.read_csv("C:\\Users\\Sananya Roy\\Downloads\\Play Store Data (1).csv")
df


# In[12]:


print(df.isnull().sum())


# In[13]:


df = df.dropna(subset=['App', 'Category', 'Installs', 'Type', 'Price', 'Content Rating', 'Android Ver', 'Size'])
df


# In[14]:


df['Installs']=df['Installs'].str.replace(',','').str.replace('+','').astype(int)
df['Price']=df['Price'].str.replace('$','').astype(float)
df['Revenue']=df['Price']*df['Installs']
df


# In[15]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
df['Size']=df['Size'].apply(convert_size)
df


# In[16]:


df['Android Ver'] = df['Android Ver'].str.extract(r'(\d+\.?\d*)').astype(float)
df


# In[17]:


filtered = df[(df['Installs'] >= 10000) & (df['Revenue'] >= 10000) & (df['Android Ver'] > 4.0) & (df['Size'] > 15) & (df['Content Rating'] == 'Everyone') & (df['App'].str.len() <= 30)]
filtered


# In[18]:


top_categories = filtered['Category'].value_counts().nlargest(3).index.tolist()
top_categories


# In[19]:


top_df = filtered[filtered['Category'].isin(top_categories)]
top_df


# In[20]:


grouped = df.groupby(['Category', 'Type']).agg({'Installs': 'mean','Revenue': 'mean'}).reset_index()
top_categories = df['Category'].value_counts().nlargest(3).index.tolist()
grouped = grouped[grouped['Category'].isin(top_categories)]
grouped


# In[21]:


import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pytz
import io
import base64
import plotly.graph_objects as go
def is_time_allowed():
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
    return 13 <= now_ist.hour < 14
def home():
    if is_time_allowed():
        return render_template_string("""<h2>Installs vs Revenue</h2><img src="/plot.png" alt="Graph">""")
    else:
        return "<h3>Graph is only visible between 1 PM and 2 PM IST.</h3>"
def plot_png():
    fig2=go.Figure()
    fig2, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for app_type, color1, color2 in [('Paid', 'blue', 'black'), ('Free', 'red', 'brown')]:
        df_type = grouped[grouped['Type'] == app_type]
        x = df_type['Category']
        y1 = df_type['Installs']
        y2 = df_type['Revenue']

        ax1.plot(x, y1, label=f'{app_type} Installs', color=color1, marker='o')
        ax2.plot(x, y2, label=f'{app_type} Revenue', color=color2, linestyle='--', marker='x')

    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Installs')
    ax2.set_ylabel('Revenue')
    ax1.set_title('Installs vs Revenue for Free vs Paid Apps by Category')
    fig2.tight_layout()
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig2)

    return f"""
    <hr>
    <h2>Installs vs Revenue (7 PM - 9 PM IST)</h2>
    <img src="data:image/png;base64,{encoded_img}" width="900">
    """
with open("dashboard.html", "r", encoding='utf-8') as f:
        wordcloud_html = f.read()
if is_time_allowed():
    chart_html=plot_png()
    final_html = wordcloud_html + chart_html
else:
    final_html = wordcloud_html + "<p style='color:red;'>chart visible only between 1 PM and 2 PM IST.</p>"
with open("dashboard.html", "w", encoding='utf-8') as f:
    f.write(final_html)
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
encoded_img = base64.b64encode(buf.read()).decode('utf-8')
buf.close()
fig2 = go.Figure()
fig2.add_layout_image(dict(source=f"data:image/png;base64,{encoded_img}",xref="paper", yref="paper",x=0, y=1, sizex=1, sizey=1,xanchor="left", yanchor="top",sizing="stretch"))
fig2.update_layout(title="Word Cloud from 5-Star Reviews (Health & Fitness)",plot_bgcolor='black',paper_bgcolor='black',font_color='white',title_font={'size':16},xaxis=dict(title_font={'size':12}),yaxis=dict(title_font={'size':12}),margin=dict(l=6,r=6,t=10,b=5))


# # TASK 3

# In[22]:


import pandas as pd
import numpy as np


# In[23]:


df=pd.read_csv("C:\\Users\\Sananya Roy\\Downloads\\Play Store Data (1).csv")
df


# In[24]:


print(df.isnull().sum())


# In[25]:


df = df.dropna(subset=['Rating'])
df


# In[26]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
df['Size']=df['Size'].apply(convert_size)
df


# In[27]:


df = df.dropna(subset=['Size', 'Last Updated'])
df


# In[28]:


df['Last Updated']=pd.to_datetime(df['Last Updated'],errors='coerce')
df


# In[29]:


df['Reviews'] = df['Reviews'].replace(',', '', regex=True).astype(int)

df['Installs'] = df['Installs'].replace('[+, ]', '', regex=True).astype(int)


# In[30]:


df=df[(df['Size'] >= 10) & (df['Last Updated'].dt.month == 1)]
df


# In[31]:


category_stats = df.groupby('Category').agg({'Rating': 'mean','Reviews': 'sum','Installs': 'sum'}).reset_index()

category_stats


# In[32]:


top_categories = category_stats[(category_stats['Rating'] >= 4.0)]
top_categories 


# In[33]:


new_df = top_categories.groupby("Category")["Installs"].sum().nlargest(10).index
new_df


# In[34]:


filtered_df=category_stats[category_stats['Category'].isin(new_df)]
filtered_df


# In[35]:


from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
def is_time_allowed():
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
    return 15 <= now_ist.hour < 17
def generate_dual_axis_chart():
    fig, ax1 = plt.subplots(figsize=(14, 6))

    categories = filtered_df['Category']
    x = range(len(categories))

    ax1.bar(x, filtered_df['Rating'], width=0.4, align='center', label='Avg Rating', color='steelblue')
    ax1.set_ylabel('Average Rating', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.bar([i + 0.4 for i in x], filtered_df['Reviews'], width=0.4, align='center', label='Total Reviews', color='orange')
    ax2.set_ylabel('Total Reviews', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.xticks([i + 0.2 for i in x], categories, rotation=45)
    plt.title('Top 10 Categories: Avg Rating vs Total Reviews')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"""
    <hr>
    <h2>Top 10 Categories: Avg Rating vs Total Reviews</h2>
    <img src="data:image/png;base64,{encoded_img}" width="900">
    """

with open("dashboard.html", "r", encoding='utf-8') as f:
    wordcloud_html = f.read()
if is_time_allowed():
    chart1_html = generate_dual_axis_chart()
    final_html = wordcloud_html + chart1_html
else:
    final_html = wordcloud_html + "<p style='color:red;'>chart visible only between 3 PM and 5 PM IST.</p>"
with open("dashboard.html", "w", encoding='utf-8') as f:
    f.write(final_html)


# # TASK 4

# In[36]:


import pandas as pd
import numpy as np
apps_df=pd.read_csv("C:\\Users\\Sananya Roy\\Downloads\\Play Store Data (1).csv")
reviews_df=pd.read_csv("C:\\Users\\Sananya Roy\\Downloads\\User Reviews.csv")
merged_df = pd.merge(apps_df, reviews_df, on='App', how='inner')
merged_df


# In[37]:


print(merged_df.isnull().sum())


# In[38]:


merged_df['Rating'] = merged_df['Rating'].fillna(merged_df['Rating'].mean())
merged_df


# In[39]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
merged_df['Size']=merged_df['Size'].apply(convert_size)
merged_df['Installs'] = merged_df['Installs'].str.replace('[+,]', '', regex=True).astype(int)
merged_df['Reviews'] = merged_df['Reviews'].replace(',', '', regex=True).astype(int)
merged_df


# In[40]:


categories=['GAME', 'BEAUTY', 'BUSINESS', 'COMICS', 'COMMUNICATION', 'DATING', 'ENTERTAIMENT', 'SOCIAL', 'EVENT']
filtered_df = merged_df[(merged_df['Rating'] > 3.5) & (merged_df['Category'].isin(categories)) & (merged_df['Reviews'] > 500) & (merged_df['App'].str.contains('s', case=False)) & (merged_df['Sentiment_Subjectivity'] > 0.5) & (merged_df['Installs'] > 50000)]
filtered_df


# In[41]:


translation = {'BEAUTY': 'सौंदर्य', 'BUSINESS': 'வணிகம்', 'DATING': 'Verabredung'}
filtered_df['Category'] = filtered_df['Category'].replace(translation)
filtered_df


# In[42]:


from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
def is_time_allowed():
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
    return 17 <= now_ist.hour < 19
def plot2_png():
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = filtered_df['Category'].unique()

    for x in categories:
        subset = filtered_df[filtered_df['Category'] == x]
        color = 'pink' if x == 'Game' else None
        scatter = ax.scatter(subset['Size'], subset['Rating'], s=subset['Installs']/20000, alpha=0.2, label=x, c=color)

    
    ax.set_xlabel("App Size (MB)")
    ax.set_ylabel("Average Rating")
    ax.set_title("App Size vs Rating Bubble Chart")

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='COMMUNICATION', markerfacecolor='red', markersize=6, alpha=0.5, linewidth=0.5),
                       Line2D([0], [0], marker='o', color='w', label='GAME', markerfacecolor='green', markersize=6, alpha=0.5, linewidth=0.5),
                       Line2D([0], [0], marker='o', color='w', label='COMICS', markerfacecolor='brown', markersize=8, alpha=0.5, linewidth=0.5),
                       Line2D([0], [0], marker='o', color='w', label='SOCIAL', markerfacecolor='pink', markersize=8, alpha=0.5, linewidth=0.5),
                       Line2D([0], [0], marker='o', color='w', label='Verabredung', markerfacecolor='purple', markersize=8, alpha=0.5, linewidth=0.5),
                       Line2D([0], [0], marker='o', color='w', label='सौंदर्ी', markerfacecolor='orange', markersize=8, alpha=0.5, linewidth=0.5),
                       Line2D([0], [0], marker='o', color='w', label='வணிகம்', markerfacecolor='yellow', markersize=8, alpha=0.5, linewidth=0.5),
                       Line2D([0], [0], marker='o', color='w', label='EVENT', markerfacecolor='blue', markersize=8, alpha=0.5, linewidth=0.5),
                       Line2D([0], [0], marker='o', color='w', label='ENTERTAINMENT', markerfacecolor='cyan', markersize=8, alpha=0.5, linewidth=0.5)]
    
    ax.legend(handles=legend_elements, title="Category", loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small', title_fontsize='medium')
    ax.set_facecolor('#ccffcc')
    fig.patch.set_facecolor('#e6e6e6')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
    fig.subplots_adjust(right=0.9)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"""
    <hr>
    <h2>App size vs Rating</h2>
    <img src="data:image/png;base64,{encoded_img}" width="900">
    """

with open("dashboard.html", "r", encoding='utf-8') as f:
    wordcloud_html = f.read()
if is_time_allowed():
    chart2_html = plot2_png()
    final_html = wordcloud_html + chart2_html
else:
    final_html = wordcloud_html + "<p style='color:red;'>chart visible only between 5 PM and 7 PM IST.</p>"
with open("dashboard.html", "w", encoding='utf-8') as f:
    f.write(final_html)


# # TASK 5

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("C:\\Users\\Sananya Roy\\Downloads\\Play Store Data (1).csv")
df


# In[3]:


print(df.isnull().sum())


# In[4]:



df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df = df[df['Installs'].str.replace('[+,]', '', regex=True).str.isnumeric()]
df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True).astype(int)
df


# In[5]:


filtered_df = df[(df['Reviews']>500) & (df['App'].str.lower().str.startswith(('x', 'y', 'z'))) & (df['App'].str.lower().str.contains('s')) & (df['Category'].str.upper().str.startswith(('E', 'C', 'B')))]
filtered_df


# In[6]:


filtered_df['Month'] = filtered_df['Last Updated'].dt.to_period('M').dt.to_timestamp()
filtered_df


# In[7]:


filtered_df[['App', 'Category', 'Last Updated', 'Installs']].dropna()
filtered_df


# In[8]:


def translated(category):
    if category== 'Beauty':
        return 'सौंदर्य'
    if category== 'Business':
        return 'வணிகம்'
    if category== 'Dating':
        return 'Partnersuche'
    else:
        return category


# In[9]:


filtered_df['Category'] = filtered_df['Category'].apply(translated)
filtered_df


# In[10]:


filtered_df['Month'] = filtered_df['Last Updated'].dt.to_period('M').dt.to_timestamp()
month=filtered_df.groupby(['Month', 'Category'])['Installs'].sum().reset_index()
month


# In[11]:


month['pct_change']=month.groupby('Category')['Installs'].pct_change()
month['pct_change']


# In[12]:


month.dropna()


# In[13]:


month['Significant_growth'] = month['pct_change']> 0.2
month['Significant_growth']


# In[14]:


get_ipython().system('pip install --upgrade kaleido')


# In[15]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
def is_time_allowed():
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
    return 18 <= now_ist.hour < 21
import plotly.io as pio 
def chart():
    month = filtered_df.groupby(['Month', 'Category'])['Installs'].sum().reset_index()
    month['Month'] = pd.to_datetime(month['Month'])
    month = month.sort_values(['Category', 'Month'])
    month['pct_change'] = month.groupby('Category')['Installs'].pct_change()
    month['Significant_growth'] = month['pct_change'] > 0.20

    fig = px.line(month, x='Month', y='Installs', color='Category', title='Total installs over time')

    for category in month['Category'].unique():
        category_df = month[month['Category'] == category].reset_index(drop=True)
        for i in range(1, len(category_df)):
            if category_df.loc[i, 'Significant_growth']:
                fig.add_trace(go.Scatter(
                    x=[category_df.loc[i-1, 'Month'], category_df.loc[i, 'Month']],
                    y=[category_df.loc[i-1, 'Installs'], category_df.loc[i, 'Installs']],
                    fill='tozeroy',
                    mode='none',
                    fillcolor='rgba(0, 100, 0, 0.3)',
                    name='>20% Growth'
                ))

    fig.update_layout(title='Monthly Installs with >20%',
                      xaxis_title='Month',
                      yaxis_title='Total Installs',
                      legend=dict(itemsizing='constant'))
    buf = io.BytesIO()
    pio.write_image(fig, buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return f"""
    <hr>
    <h2>Monthly Installs with >20% Growth</h2>
    <img src="data:image/png;base64,{encoded_img}" width="900">
    """

    
with open("dashboard.html", "r", encoding='utf-8') as f:
    wordcloud_html = f.read()
if is_time_allowed():
    chart2_html = chart()
    final_html = wordcloud_html + chart2_html
else:
    final_html = wordcloud_html + "<p style='color:red;'>chart visible only between 6 PM and 9 PM IST.</p>"
with open("dashboard.html", "w", encoding='utf-8') as f:
    f.write(final_html)


# In[ ]:




