import matplotlib.pyplot as plt
import seaborn as sns
import mpld3
from mpld3 import plugins
from mpld3 import plugins, fig_to_html
from IPython.display import display, HTML


def plot_data_matplotlib(df, x, y, title, x_label, y_label):
    
    """
    Plots arr_del15 over time using the provided pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame containing 'ds' and 'y' columns.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[x], df[y], marker='o', markersize =2, color='#3366d6')

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.title(title, fontsize=14)

    plt.grid(False)  # Remove gridlines
    sns.despine()    # Remove top and right spines
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
    
def plot_data_d3(df):
    labels = 'Delay_Arrival_Ratio'
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(df['ds'], df['arr_flights'])
    line_collections = ax.plot(df['ds'],df['y'], lw=1, alpha=.1)
    interactive_legend = plugins.InteractiveLegendPlugin(line_collections, labels)
    plugins.connect(fig, interactive_legend)
    html_string = fig_to_html(fig)  
    # Display the HTML containing the plot using IPython.display
    display(HTML(html_string))

def plot_forecast(m, forecast):
  fig = m.plot(forecast)
  plt.show() 
  
def plot_forecast_components(m, forecast):
  fig = m.plot_components(forecast)
  plt.show() 
    
